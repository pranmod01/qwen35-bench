"""Reason-then-choose scoring: the model replies freely, then the forced choice is read with that reply in context.

Opt in per model with `scoring: reason_then_choose` in the config's model block (`standard` is the default). On
every forced-choice trial the evaluator first generates a free reply (greedy, up to the model's `max_new_tokens`),
then scores the choice with the prefill "<reply>\\n\\nAnswer: " instead of "Answer: ". The prompt is unchanged, so
the only difference from standard scoring is that the model finishes whatever it wanted to say before its answer is
read off the logits. Task code is untouched: a task calls score_choices(messages) and then _call_api(messages) on
the same messages object, and the second call returns the reply already generated, so no trial is generated twice.

Per trial, `reasoning_tokens`, `reasoning_finished` and `stated_answer` (the answer the reply states, if any; a
diagnostic only, the scored answer is always the forced choice) are attached by the runner through `attach()`, and
`reasoning_metrics` adds the cell-level rates.

Why: at max_new_tokens 100 about half the 2B/4B associative-inference replies were reasoning cut off before an
answer, while the forced choice was read with no reasoning in context. This separates "cannot do it" from "was cut
off mid-thought". Ported 2026-10-29 from the RunPod zip's runpod/run_bench_ext.py (which monkey-patched the runner)
so that results/assoc_inf_reasoned (2B/4B, run on RunPod 2026-10-20) is reproducible from this repo:
`python run_bench.py configs/assoc_inf_reasoned.yaml`. Not combinable with `thinking`: both put text before the
answer, and the thinking runs are their own experiment.
"""
import re

import torch

from bench.models import CHOICE_PREFILL, Qwen35Evaluator

ANSWER_SEPARATOR = "\n\n"

# A conclusion: "the correct digit is 2", "Answer: **1**", "I choose image 2", or "Image 2 is the correct answer".
# The last one in the reply counts, and a bare "1"/"2" reply counts too. Descriptions such as "Image 1 shows a
# teddy bear" are deliberately not matched, so a reply cut off before its conclusion has no stated answer.
_CONCLUSIONS = (
    re.compile(r"\b(?:answer|digit|choice|correct|pick|choose|chose|select)\w*\s*(?:is|would\s+be)?\s*[:=]?\s*"
               r"(?:\*\*)?\s*(?:the\s+)?(?:image|word|option|number)?\s*\(?\s*(?:\*\*)?([12])\b", re.IGNORECASE),
    re.compile(r"\b([12])\b\W{0,6}(?:is|would\s+be)\s+(?:the\s+)?(?:correct|right|answer)", re.IGNORECASE),
    re.compile(r"\b(?:therefore|thus|hence|so)\b[^.\n]{0,150}?\b(?:is|would\s+be)\s+(?:\*\*)?\s*(?:the\s+)?"
               r"(?:image|word|option|number)?\s*\(?\s*(?:\*\*)?([12])\b", re.IGNORECASE),
    # "I think it is 2" / "I think that it is image 2": a hedged conclusion with none of the keywords above.
    # "think image 1 shows ..." is a description and stays unmatched (the subject must be "it").
    re.compile(r"\bthink\s+(?:that\s+)?it\s+(?:is|would\s+be)\s*(?:\*\*)?\s*(?:the\s+)?"
               r"(?:image|word|option|number)?\s*(?:\*\*)?([12])\b", re.IGNORECASE),
)


def stated_answer(reply):
    """The answer a free reply states, "1" or "2", or None if it states none."""
    # One strip over the whole set: "**1**." ends in a period that would otherwise shield the closing asterisks.
    text = (reply or "").strip("*. \t\n")
    if text in ("1", "2"):
        return text
    found = [(m.start(1), m.group(1)) for pattern in _CONCLUSIONS for m in pattern.finditer(reply or "")]
    return max(found)[1] if found else None


class ReasonThenChooseMixin:
    """The caching and bookkeeping, kept apart from the model so it can be tested without one. The class it is
    mixed into supplies _generate(messages) -> (reply, n_tokens, finished) and the standard score_choices /
    _call_api; put the mixin first so its methods wrap those."""

    trial_log = None

    def _rtc_init(self):
        self._rtc_cached = None      # (messages object, reply) of the trial being scored
        self.trial_log = []          # one entry per score_choices call, in trial order

    def score_choices(self, messages, choices=("1", "2"), prefill=CHOICE_PREFILL, thought=None):
        if thought is not None:
            raise ValueError("reason_then_choose cannot score a trial that already carries a thinking trace")
        reply, n_tokens, finished = self._generate(messages)
        self._rtc_cached = (messages, reply)
        reasoned_prefill = (reply + ANSWER_SEPARATOR + prefill) if reply else prefill
        forced = super().score_choices(messages, choices=choices, prefill=reasoned_prefill)
        self.trial_log.append({"reasoning_tokens": n_tokens, "reasoning_finished": finished,
                               "stated_answer": stated_answer(reply)})
        return forced

    def _call_api(self, messages, thought=None):
        cached = self._rtc_cached
        if cached is not None and cached[0] is messages:
            self._rtc_cached = None
            return cached[1]
        return super()._call_api(messages, thought=thought)


class ReasonThenChooseEvaluator(ReasonThenChooseMixin, Qwen35Evaluator):
    def __init__(self, **kwargs):
        if kwargs.get("thinking"):
            raise ValueError("scoring: reason_then_choose and thinking cannot be combined in one model block")
        super().__init__(**kwargs)
        self._rtc_init()

    def _generate(self, messages):
        """Qwen35Evaluator._call_api, also returning the token count and whether the reply ended by itself."""
        inputs = self._prepare_inputs(messages)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            )
        new = out[0, inputs.input_ids.shape[1]:]
        n_tokens = int(new.shape[0])
        finished = n_tokens < self.max_new_tokens or (n_tokens > 0 and int(new[-1]) in self.eos_token_id)
        reply = self.processor.batch_decode(new.unsqueeze(0), skip_special_tokens=True)[0]
        return reply.strip(), n_tokens, finished


def reasoning_metrics(trials):
    n = len(trials)
    if not n or "reasoning_tokens" not in trials[0]:
        return {}
    stated = [t for t in trials if t.get("stated_answer") is not None]
    return {
        "mean_reasoning_tokens": sum(t["reasoning_tokens"] for t in trials) / n,
        "frac_reasoning_cut_off": sum(not t["reasoning_finished"] for t in trials) / n,
        "frac_stated_answer": len(stated) / n,
        # Of the trials whose reply states an answer, how often the forced choice read after it agrees.
        "stated_forced_agreement": (sum(str(t["stated_answer"]) == str(t.get("choice", t["reported"]))
                                        for t in stated) / len(stated)) if stated else None,
    }


def reset(evaluator):
    """Call before a task runs. No-op for a standard evaluator."""
    log = getattr(evaluator, "trial_log", None)
    if log is not None:
        log.clear()


def attach(evaluator, result):
    """Call after a task ran: puts each trial's reasoning stats on the trial and the rates in the cell metrics.
    No-op for a standard evaluator or a result without trials (a skip)."""
    log = getattr(evaluator, "trial_log", None)
    trials = result.get("trials") or []
    if log is None or not trials:
        return
    if len(log) != len(trials):
        raise RuntimeError(f"{len(log)} scored choices for {len(trials)} trials; "
                           "this task does not score exactly one choice per trial")
    for trial, extra in zip(trials, log):
        trial.update(extra)
    result.setdefault("metrics", {}).update(reasoning_metrics(trials))
