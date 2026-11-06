"""Checks for bench/reasoned.py that need no model: the stated-answer reader, the reply cache between
score_choices and _call_api, and the runner hooks. Run from the repo root with the venv python:
python -m bench.test_reasoned"""
from bench import reasoned
from bench.reasoned import ReasonThenChooseMixin, stated_answer


def test_stated_answer():
    cases = {
        "2": "2",
        "**1**.": "1",
        "The correct answer is 2.": "2",
        "Answer: **1**": "1",
        "I choose image 2 because it matches.": "2",
        "Image 1 is the correct one.": "1",
        "Both were studied; therefore the answer would be image 2.": "2",
        "I think it is 2": "2",
        "I think image 1 shows a bear and image 2 a cup, and": None,
        # Descriptions are not conclusions: a reply cut off before its answer states none.
        "Image 1 shows a teddy bear and image 2 shows a cup, and the cue": None,
        "": None,
        None: None,
        # The last conclusion wins.
        "At first I would pick 1. On reflection the correct answer is 2.": "2",
    }
    for reply, want in cases.items():
        got = stated_answer(reply)
        assert got == want, (reply, got, want)


class StubBase:
    """What Qwen35Evaluator supplies, minus the model."""

    def __init__(self):
        self.calls = []

    def score_choices(self, messages, choices=("1", "2"), prefill="Answer: "):
        self.calls.append(("score", prefill))
        return {"choice": "2", "prefill": prefill}

    def _call_api(self, messages, thought=None):
        self.calls.append(("generate", thought))
        return "fresh reply"


class Stub(ReasonThenChooseMixin, StubBase):
    def __init__(self, reply):
        super().__init__()
        self._rtc_init()
        self.reply = reply

    def _generate(self, messages):
        return self.reply, 7, True


def test_reply_is_scored_then_returned_once():
    ev = Stub("I think it is 2")
    messages = [{"role": "user", "content": "which one?"}]
    forced = ev.score_choices(messages)
    assert forced["prefill"] == "I think it is 2\n\nAnswer: "
    # The task's follow-up free call on the same messages gets the cached reply, not a second decode.
    assert ev._call_api(messages) == "I think it is 2"
    assert ("generate", None) not in ev.calls
    # Only once: the next call, and a call on a different messages object, go to the model.
    assert ev._call_api(messages) == "fresh reply"
    ev.score_choices(messages)
    assert ev._call_api([{"role": "user", "content": "other"}]) == "fresh reply"
    assert ev.trial_log == [{"reasoning_tokens": 7, "reasoning_finished": True, "stated_answer": "2"}] * 2


def test_empty_reply_keeps_plain_prefill():
    ev = Stub("")
    assert ev.score_choices([{}])["prefill"] == "Answer: "
    assert ev.trial_log[0]["stated_answer"] is None


def test_refuses_thinking_trace():
    try:
        Stub("x").score_choices([{}], thought={"prefill": "..."})
        raise AssertionError("a thinking trace should be refused")
    except ValueError:
        pass


def test_runner_hooks():
    ev = Stub("the answer is 1")
    messages = [{}]
    ev.score_choices(messages)
    ev._call_api(messages)
    result = {"trials": [{"reported": "2"}], "metrics": {"accuracy": 0.0}}
    reasoned.attach(ev, result)
    assert result["trials"][0]["stated_answer"] == "1" and result["trials"][0]["reasoning_tokens"] == 7
    m = result["metrics"]
    assert m["frac_stated_answer"] == 1.0 and m["stated_forced_agreement"] == 0.0 and m["frac_reasoning_cut_off"] == 0.0
    # A task that scores a different number of choices than it has trials is a bug, not a metric.
    ev.score_choices(messages)
    try:
        reasoned.attach(ev, {"trials": [{"reported": "1"}, {"reported": "1"}, {"reported": "2"}], "metrics": {}})
        raise AssertionError("mismatched trial count should raise")
    except RuntimeError:
        pass
    reasoned.reset(ev)
    assert ev.trial_log == []
    # Standard evaluators and skipped cells pass through untouched.
    plain = object()
    reasoned.reset(plain)
    skipped = {"skipped": "too many images"}
    reasoned.attach(plain, skipped)
    reasoned.attach(ev, skipped)
    assert skipped == {"skipped": "too many images"}


if __name__ == "__main__":
    tests = [test_stated_answer, test_reply_is_scored_then_returned_once, test_empty_reply_keeps_plain_prefill,
             test_refuses_thinking_trace, test_runner_hooks]
    for test in tests:
        test()
    print(f"all {len(tests)} checks passed")
