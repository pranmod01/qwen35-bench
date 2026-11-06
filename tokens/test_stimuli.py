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
from bench.tasks import afc_recognition_eval, pam_eval, serial_order_eval, token_recall_eval
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
    # L=1: single-token items work with novel foils; a recombined foil would be half of nothing.
    t = token_stimuli.recognition_trials(pool, 10, 1, "novel")
    assert all(len(item.ids) == 1 for item in t["study_sequence"])
    for bad in (lambda: token_stimuli.recognition_trials(pool, 10, 1, "recombined"),
                lambda: token_stimuli.recall_trials(pool, 10, 1)):
        try:
            bad()
            raise AssertionError("item_length 1 should be refused")
        except ValueError:
            pass


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
    # L=1 is allowed for novel foils and serial order, refused where half an item is needed.
    l1 = {**cfg, "datasets": ["tokens"], "foil_types": ["novel"], "item_lengths": [1], "token_pools": ["clean"]}
    assert afc_recognition_eval.max_n_images(afc_recognition_eval.expand_conditions(l1)[0]) == 245450 - 1
    for bad in (lambda: afc_recognition_eval.expand_conditions({**l1, "foil_types": ["recombined"]}),
                lambda: token_recall_eval.expand_conditions({"task": "token_recall", "datasets": ["tokens"],
                                                             "n_images": [5], "item_lengths": [1], "token_pools": ["clean"]})):
        try:
            bad()
            raise AssertionError("item_length 1 should be refused")
        except ValueError:
            pass


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


def test_pam_trials(pool):
    random.seed(7)
    words = pam_eval.load_wordpool("was")
    positions = set()
    for _ in range(200):
        t = token_stimuli.pam_trials(pool, 10, 8, words)
        items, paired = zip(*t["study_sequence"])
        assert len({i for item in items for i in item.ids}) == 10 * 8   # no token reused within an episode
        assert len(set(paired)) == 10 and set(paired) <= set(words)
        test = t["test_phase"][0]
        k = test["metadata"]["test_position"]
        assert test["image"] == items[k - 1] and test["target"] == paired[k - 1]
        assert test["metadata"]["cue_ids"] == list(items[k - 1].ids)
        positions.add(k)
    assert positions == set(range(1, 11)), positions
    try:
        token_stimuli.pam_trials(pool, len(words) + 1, 8, words)
        raise AssertionError("more pairs than words should be refused")
    except ValueError:
        pass


def test_pam_messages(ev, pool):
    """Items spliced verbatim in the sandwich layout, the word on both sides of each, the prompt reworded."""
    random.seed(8)
    t = token_stimuli.pam_trials(pool, 5, 8, pam_eval.load_wordpool("was"))
    test = t["test_phase"][0]
    messages = pam_eval.build_messages(ev, t["study_sequence"], t["study_prompt"], test["image"], layout="sandwich")
    ids = ev._prepare_inputs(messages)["input_ids"][0].tolist()
    for item, word in t["study_sequence"]:
        starts = [k for k in range(len(ids)) if tuple(ids[k:k + 8]) == item.ids]
        assert len(starts) == (2 if item == test["image"] else 1), (word, starts)
        # The item is bracketed by its own word: "Word: W\n<item>\nWord: W".
        before = ev.processor.tokenizer.decode(ids[max(0, starts[0] - 8):starts[0]])
        after = ev.processor.tokenizer.decode(ids[starts[0] + 8:starts[0] + 16])
        assert before.endswith(f"Word: {word}\n"), before
        assert after.startswith(f"\nWord: {word}"), after
    text = ev.processor.tokenizer.decode(ids)
    assert "Remember the word paired with each item." in text
    assert "I have studied the item-pairs." in text
    assert "which word was shown with this item? Do not name or describe the item." in text
    assert "image" not in text.lower() and "<<<tokens" not in text
    # The cue closes the prompt's last user turn, followed by the question on its own line.
    cue = max(k for k in range(len(ids)) if tuple(ids[k:k + 8]) == test["image"].ids)
    assert ev.processor.tokenizer.decode(ids[cue + 8:cue + 12]).startswith("\nIn the")
    for layout in ("original", "pair_numbered"):
        try:
            pam_eval.build_messages(ev, t["study_sequence"], t["study_prompt"], test["image"], layout=layout)
            raise AssertionError(f"{layout} should be refused for token items")
        except ValueError:
            pass


def test_pam_config_expansion():
    cfg = {"task": "paired_associates", "datasets": ["Brady2008", "tokens"], "prompts": ["sandwich"],
           "n_images": [5], "item_lengths": [8, 256], "token_pools": ["clean"]}
    conds = pam_eval.expand_conditions(cfg)
    assert len(conds) == 3, conds
    brady = next(c for c in conds if c["dataset"] == "Brady2008")
    assert "item_length" not in brady and pam_eval.max_n_images(brady) == 1638
    tok8 = next(c for c in conds if c["dataset"] == "tokens" and c["item_length"] == 8)
    tok256 = next(c for c in conds if c["dataset"] == "tokens" and c["item_length"] == 256)
    assert pam_eval.max_n_images(tok8) == 1638                       # the WAS pool binds before the token pool
    assert pam_eval.max_n_images(tok256) == 245450 // 256            # 958: the token pool binds
    assert pam_eval.max_n_images({**tok8, "wordpool": "extended"}) == 4000
    assert pam_eval.min_n_images(tok8) == 1
    for bad in ({**cfg, "prompts": ["original"]}, {**cfg, "prompts": ["sandwich", "original"]},
                {k: v for k, v in cfg.items() if k != "item_lengths"}):
        try:
            pam_eval.expand_conditions(bad)
            raise AssertionError(f"should be refused: {bad}")
        except ValueError:
            pass


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
    scored = Qwen35Evaluator.score_tail(copier, prompt, t["cue"], t["target_tail"])
    # Off by one in either direction compares each prediction with the wrong token, and nothing matches.
    assert scored["top1"] == [True] * 4, scored
    assert all(lp > -0.01 for lp in scored["logprobs"]), scored


class KnownTokenModel:
    """Stand-in model that predicts the next input token only when it is in `known`; a model that holds
    the studied tokens and nothing else."""
    def __init__(self, known):
        self.known = set(known)

    def __call__(self, input_ids, logits_to_keep, **_):
        kept = input_ids[0, -logits_to_keep:]
        logits = torch.zeros(1, logits_to_keep, 300000)
        for j in range(logits_to_keep - 1):
            if int(kept[j + 1]) in self.known:
                logits[0, j, kept[j + 1]] = 30.0
        return SimpleNamespace(logits=logits)


def test_copy_prompt(ev, pool):
    """The copy test turn names items, not images; the options are scored verbatim after the generation prompt,
    and a model that holds the studied tokens picks the studied option whichever side it is on."""
    random.seed(7)
    for target in (1, 2):
        while True:
            t = token_stimuli.recognition_trials(pool, 5, 8, "novel")
            test = t["test_phase"][0]
            if test["target"] == target:
                break
        messages = afc_recognition_eval.build_messages(ev, t["study_sequence"], t["study_prompt"],
                                                       test["images"], "copy")
        text = ev.processor.tokenizer.decode(ev._prepare_inputs(messages)["input_ids"][0].tolist())
        assert "image" not in text.lower() and "Reply with that item exactly" in text, text[-400:]
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), text[-200:]
        studied = {i for item in t["study_sequence"] for i in item.ids}
        holder = SimpleNamespace(model=KnownTokenModel(studied), processor=ev.processor, thinking=None,
                                 _prepare_inputs=ev._prepare_inputs, score_tail=None)
        holder.score_tail = lambda prompt, cue, tail: Qwen35Evaluator.score_tail(holder, prompt, cue, tail)
        forced = afc_recognition_eval.score_copy(holder, messages, test["images"])
        assert int(forced["choice"]) == target, (target, forced)
        assert len(forced["option_logprobs"]["1"]) == 8 and len(forced["option_logprobs"]["2"]) == 8
        assert forced["probs"][str(target)] > 0.99, forced["probs"]
        # The free-text check accepts only an exact reproduction of one option.
        tok = ev.processor.tokenizer
        assert afc_recognition_eval.parse_copy_response(ev, tok.decode(list(test["images"][target - 1].ids)) + "\n",
                                                        test["images"]) == target
        assert afc_recognition_eval.parse_copy_response(ev, "I think it was the first one", test["images"]) == -1


def test_copy_config_expansion():
    cfg = {"task": "afc_recognition", "datasets": ["tokens"], "foil_types": ["novel"], "single_foil": True,
           "n_images": [5], "prompts": ["copy"], "item_lengths": [8], "token_pools": ["clean"]}
    assert len(afc_recognition_eval.expand_conditions(cfg)) == 1
    try:
        afc_recognition_eval.expand_conditions({**cfg, "datasets": ["Brady2008"]})
        raise AssertionError("copy prompt on images should be refused")
    except ValueError:
        pass


def synthetic_corpus(path, n_docs=12, paragraphs_per_doc=6, seed=0):
    """A small corpus in the fetch_wikitext layout: digit-free paragraphs of short sentences, one document of
    paragraphs with numbers that must never yield a span."""
    rng = random.Random(seed)
    words = ("river stone cloud window garden letter silver market harbour candle meadow thunder orchard "
             "bridge lantern saddle velvet copper willow anchor").split()
    verbs = "crossed held carried watched followed painted opened measured".split()
    def sentence():
        return (f"{rng.choice(words).capitalize()} {rng.choice(verbs)} the {rng.choice(words)} "
                f"beside the {rng.choice(words)} {rng.choice(words)}.")
    lines = []
    for d in range(n_docs):
        lines += [f"= Document {d} =", "", "= = Section = =", ""]
        for _ in range(paragraphs_per_doc):
            lines += [" ".join(sentence() for _ in range(rng.randint(2, 5))), ""]
    lines += ["= Numbers =", ""] + ["The year 1987 saw 12 events. Then 3 more came in 1990."] * 4
    path.write_text("\n".join(lines) + "\n")


def test_natural_pools(ev):
    """Spans are clean sentence starts cut at exactly L tokens, scrambled spans are their permutations, and the
    trial builders and splicing work unchanged on them."""
    import tempfile
    from tokens import corpus
    tok = ev.processor.tokenizer
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_corpus(tmp / "train.txt")
        docs = list(corpus.read_paragraphs(tmp / "train.txt"))
        assert len(docs) == 12 * 6 + 4 and max(d for d, _ in docs) == 12
        for L in (4, 8):
            meta = corpus.build_cache(tok, L, corpus=tmp / "train.txt", cache_dir=tmp / "cache", target=40, minimum=10)
            assert meta["n_spans"] == 40 and meta["item_length"] == L and meta["n_documents"] == 13
        # The target is a cap: a smaller corpus gives what it has, down to the minimum.
        meta = corpus.build_cache(tok, 4, corpus=tmp / "train.txt", cache_dir=tmp / "cache", target=1000, minimum=10)
        assert 40 < meta["n_spans"] < 1000 and meta["paragraphs_tried"] == 12 * 6 + 4, meta
        try:
            corpus.build_cache(tok, 8, corpus=tmp / "train.txt", cache_dir=tmp / "cache", target=1000)
            raise AssertionError("a corpus below the minimum should be refused")
        except ValueError:
            pass
        corpus.build_cache(tok, 4, corpus=tmp / "train.txt", cache_dir=tmp / "cache", target=40, minimum=10)
        natural = corpus.SpanPool("natural", tok, cache_dir=tmp / "cache")
        scrambled = corpus.SpanPool("scrambled", tok, cache_dir=tmp / "cache")
        spans = natural.spans(8)
        assert spans.shape == (40, 8) and len({tuple(s) for s in spans}) == 40
        for span in spans:
            text = tok.decode([int(i) for i in span])
            assert text[0].isupper() and text == text.strip() and not any(c.isdigit() for c in text), text
            assert "1987" not in text and "\n" not in text
            # Cut at a sentence start, so the span is exactly what the tokenizer gives for that text.
            assert tok.encode(text, add_special_tokens=False)[:7] == [int(i) for i in span[:7]], text
        # A cut mid-paragraph leaves the first L tokens as in the full tokenisation.
        para = "The " + " ".join(["harbour lantern"] * 400) + "."   # long enough for cut_span to truncate
        full = tok.encode(para, add_special_tokens=False)
        assert corpus.cut_span(tok, para, 0, 64) == full[:64]
        assert corpus.cut_span(tok, "Too short.", 0, 8) is None
        assert corpus.cut_span(tok, "The year 1987 was long. " * 8, 0, 8) is None   # digits are never in a span

        random.seed(11)
        items = token_stimuli.sample_items(natural, 10, 8)
        assert len({item.ids for item in items}) == 10 and all(len(item) == 8 for item in items)
        random.seed(11)
        shuffled = token_stimuli.sample_items(scrambled, 10, 8)
        assert all(sorted(a.ids) == sorted(b.ids) for a, b in zip(items, shuffled))
        assert any(a.ids != b.ids for a, b in zip(items, shuffled))
        # Seeds reproduce; pools report their size before a tokenizer loads.
        random.seed(12); a = token_stimuli.recognition_trials(natural, 5, 8, "recombined")
        random.seed(12); b = token_stimuli.recognition_trials(natural, 5, 8, "recombined")
        assert a == b
        old = corpus.CACHE_DIR
        corpus.CACHE_DIR = tmp / "cache"
        try:
            assert token_stimuli.max_items("natural", 8) == 40 and token_stimuli.max_items("scrambled", 4) == 40
            for bad in (lambda: token_stimuli.max_items("natural", 16), lambda: natural.spans(16)):
                try:
                    bad()
                    raise AssertionError("a missing cache should be refused with the build command")
                except FileNotFoundError as e:
                    assert "python -m tokens.corpus --lengths 16" in str(e)
        finally:
            corpus.CACHE_DIR = old
        try:
            token_stimuli.sample_items(natural, 41, 8)
            raise AssertionError("more items than spans should be refused")
        except ValueError:
            pass
        # The builders and the splice path are shared with the random pools.
        t = token_stimuli.recall_trials(natural, 5, 8)
        assert len(t["cue"]) == 4 and t["cue"] + t["target_tail"] == t["study_sequence"][t["metadata"]["test_position"] - 1].ids
        t = token_stimuli.serial_order_trials(scrambled, 12, 4)
        assert all(len(item) == 4 for item in t["study_sequence"])
        random.seed(13)
        t = token_stimuli.recognition_trials(natural, 5, 8, "novel")
        test = t["test_phase"][0]
        messages = afc_recognition_eval.build_messages(ev, t["study_sequence"], t["study_prompt"], test["images"], "original")
        ids = ev._prepare_inputs(messages, prefill="Answer: ")["input_ids"][0].tolist()
        correct = test["images"][test["target"] - 1]
        for item in t["study_sequence"] + test["images"]:
            starts = [k for k in range(len(ids)) if tuple(ids[k:k + 8]) == item.ids]
            assert len(starts) == (2 if item == correct else 1), starts   # the tested item is studied and an option
            assert all(tok.decode(ids[k - 1:k]).endswith("\n") for k in starts)   # each after the item separator
        # A tokenizer with another vocabulary cannot use the cache.
        class Other:
            name_or_path = "other"
            def __len__(self):
                return 7
        other = Other()
        try:
            corpus.SpanPool("natural", other, cache_dir=tmp / "cache").spans(8)
            raise AssertionError("a vocabulary-size mismatch should be refused")
        except ValueError:
            pass


def test_detokenize():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from fetch_wikitext import detokenize
    assert detokenize("The vote , held in 1987 , didn 't pass ( again ) ; it 's over .") == \
        "The vote, held in 1987, didn't pass (again); it's over."
    assert detokenize('A 3 @.@ 5 km long @-@ distance race with 1 @,@ 200 runners . He said " no " .') == \
        'A 3.5 km long-distance race with 1,200 runners. He said "no".'


def test_baseline_trials(pool):
    """Baseline trials cue or test an unstudied item, so nothing in the choice can come from the study list."""
    random.seed(9)
    for _ in range(100):
        t = token_stimuli.recognition_trials(pool, 10, 8, "recombined", baseline=True)
        study, test = t["study_sequence"], t["test_phase"][0]
        studied = {i for item in study for i in item.ids}
        correct, foil = test["images"][test["target"] - 1], test["images"][2 - test["target"]]
        assert not set(correct.ids) & studied and not set(foil.ids) & studied and not set(foil.ids) & set(correct.ids)
        assert test["metadata"]["test_position"] is None and test["metadata"]["foil_head_position"] is None
        r = token_stimuli.recall_trials(pool, 10, 8, baseline=True)
        studied = {i for item in r["study_sequence"] for i in item.ids}
        assert not set(r["cue"] + r["target_tail"] + r["foil_tail"]) & studied
        assert len(r["cue"]) == 4 and r["metadata"]["test_position"] is None and len(r["study_sequence"]) == 10
    try:
        token_stimuli.recognition_trials(pool, 10, 8, "novel", baseline=True)
        raise AssertionError("a novel baseline (two unstudied intact items) should be refused")
    except ValueError:
        pass


def test_baseline_config_expansion():
    cfg = {"task": "afc_recognition", "datasets": ["tokens"], "foil_types": ["recombined"], "n_images": [5],
           "prompts": ["original"], "single_foil": True, "item_lengths": [8], "token_pools": ["clean"], "baseline": True}
    conds = afc_recognition_eval.expand_conditions(cfg)
    assert len(conds) == 1 and conds[0]["baseline"] is True
    assert afc_recognition_eval.max_n_images(conds[0]) == 245450 // 8 - 3
    assert "baseline" not in afc_recognition_eval.expand_conditions({**cfg, "baseline": False})[0]
    for bad in ({**cfg, "foil_types": ["novel"]}, {**cfg, "datasets": ["Brady2008"]}):
        try:
            afc_recognition_eval.expand_conditions(bad)
            raise AssertionError(f"should be refused: {bad}")
        except ValueError:
            pass
    rc = token_recall_eval.expand_conditions({"task": "token_recall", "datasets": ["tokens"], "n_images": [5],
                                              "item_lengths": [8], "token_pools": ["clean"], "baseline": True})
    assert rc[0]["baseline"] is True and token_recall_eval.max_n_images(rc[0]) == 245450 // 8 - 2


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
        lambda: test_copy_prompt(ev, pool),
        test_copy_config_expansion,
        lambda: test_pam_trials(pool),
        lambda: test_pam_messages(ev, pool),
        test_pam_config_expansion,
        lambda: test_natural_pools(ev),
        test_detokenize,
        lambda: test_baseline_trials(pool),
        test_baseline_config_expansion,
    ]
    for test in tests:
        test()
    print(f"all {len(tests)} checks passed")
