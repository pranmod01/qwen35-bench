"""Checks for the token-item stimuli and their splicing into prompts. CPU only; needs the tokenizer in HF_HOME.

    HF_HOME=$ZROOT/hf_cache HF_HUB_OFFLINE=1 python tokens/test_stimuli.py
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoProcessor

from bench.models import Qwen35Evaluator
from bench.tasks import afc_recognition_eval, serial_order_eval, token_recall_eval
from tokens import stimuli as token_stimuli


def make_evaluator():
    # Only the processor is needed to build inputs, so skip loading weights.
    ev = Qwen35Evaluator.__new__(Qwen35Evaluator)
    ev.processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
    ev.model = SimpleNamespace(device="cpu")
    ev.max_image_size = 512
    ev.thinking = None
    return ev


def test_pools(tok):
    for name in token_stimuli.POOL_SIZES:
        pool = token_stimuli.build_pool(tok, name)  # raises if the size drifted
        decoded = [tok.decode([i]) for i in random.sample(pool, 2000)]
        assert not any(c.isdigit() or c == "\n" for s in decoded for c in s), name
    # alpha is a subset of clean
    assert set(token_stimuli.build_pool(tok, "alpha")) <= set(token_stimuli.build_pool(tok, "clean"))


def test_splice_matches_processor_without_items(ev):
    """The chunked tokenization must agree with the processor wherever there is nothing to splice."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Here is a sequence of items to remember."}]},
        {"role": "assistant", "content": "I have studied the sequence of items."},
        {"role": "user", "content": [{"type": "text", "text": afc_recognition_eval.PROMPTS["original"]}]},
    ]
    text = ev.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + "Answer: "
    spliced = ev._splice_token_items(text, [])
    reference = ev.processor(text=[text], return_tensors="pt")
    assert spliced.keys() == reference.keys(), (spliced.keys(), reference.keys())
    for key in reference:
        assert spliced[key].tolist() == reference[key].tolist(), key


def test_items_spliced_verbatim(ev, pool):
    random.seed(0)
    trial = token_stimuli.recognition_trials(pool, 5, 16, "novel")
    test = trial["test_phase"][0]
    messages = afc_recognition_eval.build_messages(ev, trial["study_sequence"], trial["study_prompt"],
                                                   test["images"], "original")
    ids = ev._prepare_inputs(messages, prefill="Answer: ")["input_ids"][0].tolist()
    items = trial["study_sequence"] + test["images"]
    # Every item occurs as an exact contiguous run of its IDs: 5 study slots + 2 test slots.
    starts = {k for item in items for k in range(len(ids)) if tuple(ids[k:k + 16]) == item.ids}
    assert len(starts) == len(items), starts
    text = ev.processor.tokenizer.decode(ids)
    assert "Here is a sequence of items to remember." in text
    assert "Which of these two items was in the study sequence?" in text
    assert "image" not in text.lower()
    assert "<<<tokens" not in text


def test_recognition_trials(pool):
    random.seed(1)
    positions = set()
    for _ in range(200):
        for foil_type in ("novel", "recombined"):
            t = token_stimuli.recognition_trials(pool, 10, 8, foil_type)
            study = t["study_sequence"]
            test = t["test_phase"][0]
            correct, foil = test["images"][test["target"] - 1], test["images"][2 - test["target"]]
            studied_tokens = {i for item in study for i in item.ids}
            assert len(studied_tokens) == 10 * 8  # no token reused within an episode
            assert correct == study[test["metadata"]["test_position"] - 1]
            assert foil not in study
            if foil_type == "novel":
                assert not set(foil.ids) & studied_tokens
                positions.add(test["metadata"]["test_position"])
            else:
                m = test["metadata"]
                assert set(foil.ids) <= studied_tokens
                assert foil.ids[:4] == study[m["foil_head_position"] - 1].ids[:4]
                assert foil.ids[4:] == study[m["foil_tail_position"] - 1].ids[4:]
                assert not set(foil.ids) & set(correct.ids)
    assert positions == set(range(1, 11)), positions  # tests every list position, not just the first


def test_serial_order_trials(pool):
    random.seed(2)
    for _ in range(200):
        t = token_stimuli.serial_order_trials(pool, 12, 8)
        test = t["test_phase"][0]
        m = test["metadata"]
        left = t["study_sequence"].index(test["images"][0]) + 1
        right = t["study_sequence"].index(test["images"][1]) + 1
        assert (left, right) == (m["left_image_serial_position"], m["right_image_serial_position"])
        assert test["target"] == (1 if left < right else 2)


def test_pair_sampler_matches_reference():
    """sample_pair_by_lag must draw pairs with balanced_sample_by_lag's distribution (one test).

    That distribution is: a lag bin uniformly, then a pair uniformly within it, so a lag d is drawn in
    proportion to its n - d pairs and, given d, every start position is equally likely. Both samplers are
    checked against those exact values.
    """
    from itertools import combinations
    from tasks.serial_order_memory import balanced_sample_by_lag
    n, draws = 120, 40000  # n > 101, so all five lag bins are non-empty
    pairs = list(combinations(range(n), 2))
    random.seed(4)
    samplers = {"ours": lambda: token_stimuli.sample_pair_by_lag(n),
                "reference": lambda: balanced_sample_by_lag(pairs, 1)[0]}
    for name, sample in samplers.items():
        drawn = [sample() for _ in range(draws)]
        assert all(0 <= i < j < n for i, j in drawn), name
        for lo, hi in token_stimuli.LAG_BINS:
            hi = hi or n - 1
            in_bin = [(i, j) for i, j in drawn if lo <= j - i <= hi]
            assert abs(len(in_bin) / draws - 0.2) < 0.01, (name, lo, len(in_bin))
            weights = {d: n - d for d in range(lo, hi + 1)}
            exact_mean_lag = sum(d * w for d, w in weights.items()) / sum(weights.values())
            mean_lag = sum(j - i for i, j in in_bin) / len(in_bin)
            assert abs(mean_lag - exact_mean_lag) < max(0.02 * exact_mean_lag, 0.05), (name, lo, mean_lag)
            # Start position given the lag, as a fraction of its range: uniform has mean 0.5.
            spread = [i / (n - (j - i) - 1) for i, j in in_bin if n - (j - i) > 1]
            assert abs(sum(spread) / len(spread) - 0.5) < 0.02, (name, lo)


def test_seeded(pool):
    random.seed(3)
    a = token_stimuli.recognition_trials(pool, 20, 8, "recombined")
    random.seed(3)
    b = token_stimuli.recognition_trials(pool, 20, 8, "recombined")
    assert a == b


def test_config_expansion():
    cfg = {"task": "afc_recognition", "datasets": ["Brady2008", "tokens"], "foil_types": ["novel", "recombined"],
           "n_images": [5], "prompts": ["original"], "single_foil": True,
           "item_lengths": [8, 64], "token_pools": ["alpha"]}
    conds = afc_recognition_eval.expand_conditions(cfg)
    # Brady: novel only; tokens: 2 foil types x 2 lengths
    assert len(conds) == 5, conds
    assert all("item_length" not in c for c in conds if c["dataset"] == "Brady2008")
    tok64 = next(c for c in conds if c["dataset"] == "tokens" and c["item_length"] == 64 and c["foil_type"] == "novel")
    assert afc_recognition_eval.max_n_images(tok64) == 104164 // 64 - 1
    try:
        afc_recognition_eval.expand_conditions({**cfg, "single_foil": False})
        raise AssertionError("tokens without single_foil should be refused")
    except ValueError:
        pass
    so = serial_order_eval.expand_conditions({"task": "serial_order", "datasets": ["tokens"], "n_images": [5],
                                              "prompts": ["original"], "item_lengths": [64], "token_pools": ["clean"]})
    assert serial_order_eval.max_n_images(so[0]) == 245450 // 64


def test_recall_trials(pool):
    random.seed(5)
    positions = set()
    for _ in range(200):
        t = token_stimuli.recall_trials(pool, 10, 8)
        study = t["study_sequence"]
        tested = study[t["metadata"]["test_position"] - 1]
        assert t["cue"] + t["target_tail"] == tested.ids
        assert len(t["foil_tail"]) == 4 and not set(t["foil_tail"]) & {i for item in study for i in item.ids}
        positions.add(t["metadata"]["test_position"])
    assert positions == set(range(1, 11)), positions


class NextTokenModel:
    """Stand-in model whose top prediction at every position is the next input token, i.e. a perfect copier."""
    def __call__(self, input_ids, logits_to_keep, **_):
        kept = input_ids[0, -logits_to_keep:]
        logits = torch.zeros(1, logits_to_keep, 300000)
        for j in range(logits_to_keep - 1):
            logits[0, j, kept[j + 1]] = 30.0
        return SimpleNamespace(logits=logits)


def test_recall_scoring(ev, pool):
    """Studied items spliced verbatim, and score_tail reads the logits that predict the tail, not one off."""
    random.seed(6)
    t = token_stimuli.recall_trials(pool, 5, 8)
    prompt = ev._prepare_inputs(token_recall_eval.build_messages(ev, t))
    ids = prompt["input_ids"][0].tolist()
    assert all(any(tuple(ids[k:k + 8]) == item.ids for k in range(len(ids))) for item in t["study_sequence"])
    assert ev.processor.tokenizer.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    copier = SimpleNamespace(model=NextTokenModel())
    scored = token_recall_eval.score_tail(copier, prompt, t["cue"], t["target_tail"])
    # Off by one in either direction compares each prediction with the wrong token, and nothing matches.
    assert scored["top1"] == [True] * 4, scored
    assert all(lp > -0.01 for lp in scored["logprobs"]), scored


if __name__ == "__main__":
    ev = make_evaluator()
    tok = ev.processor.tokenizer
    pool = token_stimuli.build_pool(tok, "alpha")
    tests = [
        lambda: test_pools(tok),
        lambda: test_splice_matches_processor_without_items(ev),
        lambda: test_items_spliced_verbatim(ev, pool),
        lambda: test_recognition_trials(pool),
        lambda: test_serial_order_trials(pool),
        test_pair_sampler_matches_reference,
        lambda: test_seeded(pool),
        test_config_expansion,
        lambda: test_recall_trials(pool),
        lambda: test_recall_scoring(ev, pool),
    ]
    for test in tests:
        test()
    print(f"all {len(tests)} checks passed")
