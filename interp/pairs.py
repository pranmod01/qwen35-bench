"""Clean/corrupted trial pairs (INTERP_PLAN.md, section 4).

Exactly one study slot differs between the two prompts, positions stay aligned, and the correct answer flips.
Trials come from the benchmark's own builders so a pair is a benchmark episode plus its counterfactual; the
messages are rendered by the task modules' build_messages, so the tokens match what the benchmark scored.
"""
import random
from dataclasses import dataclass

from bench.models import CHOICE_PREFILL
from bench.tasks import afc_recognition_eval, pam_eval, serial_order_eval
from tasks.paired_associate_memory import PairedAssociateMemoryTask
from tasks.recognition import AFCRecognitionTask
from tasks.serial_order_memory import AFCSerialOrderMemoryTask
from tokens import stimuli as token_stimuli


@dataclass
class Pair:
    task: str
    clean: list            # messages
    corrupt: list          # messages
    answer: object         # correct answer in the clean prompt: "1"/"2", or a tuple of token ids (PAM)
    counterfactual: object # correct answer in the corrupted prompt
    slot: int              # 0-based study position that differs (serial order: the earlier of the two swapped)
    n: int
    meta: dict
    prefill: str = CHOICE_PREFILL   # None for PAM, which generates the word straight after the think block


def _index_by_identity(items, item):
    return next(i for i, x in enumerate(items) if x is item)


def aligned(evaluator, clean, corrupt, prefill):
    """True if the two rendered prompts have the same token count (positions line up slot for slot)."""
    a = evaluator._prepare_inputs(clean, prefill=prefill)["input_ids"]
    b = evaluator._prepare_inputs(corrupt, prefill=prefill)["input_ids"]
    return a.shape == b.shape


def recognition_pair(evaluator, dataset, n, item_length=8, token_pool="clean"):
    """Clean: target T studied at a random slot, foil F never shown. Corrupted: F studied in T's slot, T never shown.

    The probe shows T and F in the same order in both, so the answer flips 1 <-> 2.
    """
    if dataset == token_stimuli.DATASET:
        pool = token_stimuli.build_pool(evaluator.processor.tokenizer, token_pool)
        trial = token_stimuli.recognition_trials(pool, n, item_length, "novel")
        test = trial["test_phase"][0]
        slot = test["metadata"]["test_position"] - 1
    else:
        task = AFCRecognitionTask(dataset_name=dataset, n_images=n, foil_type="novel", single_foil=True)
        trial = task.get_trials()
        test = trial["test_phase"][0]
        slot = _index_by_identity(trial["study_sequence"], test["images"][test["target"] - 1])
    study = trial["study_sequence"]
    target = test["target"]
    foil = test["images"][2 - target]
    corrupt_study = list(study)
    corrupt_study[slot] = foil
    build = lambda seq: afc_recognition_eval.build_messages(evaluator, seq, trial["study_prompt"], test["images"], "original")
    return Pair(
        task="afc_recognition", clean=build(study), corrupt=build(corrupt_study),
        answer=str(target), counterfactual=str(3 - target), slot=slot, n=n,
        meta={"dataset": dataset, **({"item_length": item_length, "token_pool": token_pool} if dataset == token_stimuli.DATASET else {})},
    )


def serial_order_pair(evaluator, dataset, n, item_length=8, token_pool="clean"):
    """Clean: A at slot i, B at slot j > i. Corrupted: the two slots swapped. Same probe, so the answer flips 1 <-> 2."""
    if dataset == token_stimuli.DATASET:
        pool = token_stimuli.build_pool(evaluator.processor.tokenizer, token_pool)
        trial = token_stimuli.serial_order_trials(pool, n, item_length)
    else:
        trial = AFCSerialOrderMemoryTask(dataset_name=dataset, n_images=n, n_tests=1).get_trials()
    study, test = trial["study_sequence"], trial["test_phase"][0]
    i, j = sorted((test["metadata"]["left_image_serial_position"] - 1, test["metadata"]["right_image_serial_position"] - 1))
    corrupt_study = list(study)
    corrupt_study[i], corrupt_study[j] = study[j], study[i]
    build = lambda seq: serial_order_eval.build_messages(evaluator, seq, trial["study_prompt"], test["images"], "original")
    target = test["target"]
    return Pair(
        task="serial_order", clean=build(study), corrupt=build(corrupt_study),
        answer=str(target), counterfactual=str(3 - target), slot=i, n=n,
        meta={"dataset": dataset, "second_slot": j, "serial_lag": test["metadata"]["serial_lag"],
              **({"item_length": item_length, "token_pool": token_pool} if dataset == token_stimuli.DATASET else {})},
    )


def word_ids(tokenizer, word):
    """Token ids for a word as an answer (no leading space, straight after the think block) and in the study list
    (after "Word:" so with a leading space). The answer set holds the first token of each form, as the benchmark's
    digit scoring does; the study-list form fixes the token count that the counterfactual word must match."""
    bare = tokenizer.encode(word, add_special_tokens=False)
    spaced = tokenizer.encode(f" {word}", add_special_tokens=False)
    return {bare[0], spaced[0]}, len(spaced)


def pam_pair(evaluator, dataset, n, item_length=8, token_pool="clean", wordpool=None):
    """Clean: (item, WORD_B) at slot s. Corrupted: (item, WORD_R), R unstudied, same token count, different first token.

    The probe cues with the item and asks for its word; the answer flips B <-> R, scored on the first token.
    """
    tokenizer = evaluator.processor.tokenizer
    wordpool = wordpool or ("extended" if dataset == token_stimuli.DATASET else pam_eval.DEFAULT_WORDPOOL)
    words = pam_eval.load_wordpool(wordpool)
    if dataset == token_stimuli.DATASET:
        pool = token_stimuli.build_pool(tokenizer, token_pool)
        trial = token_stimuli.pam_trials(pool, n, item_length, words)
    else:
        trial = PairedAssociateMemoryTask(dataset_name=dataset, n_images=n, pair_type="word",
                                          wordpool_path=pam_eval.WORDPOOLS[wordpool]).get_trials()
    study, test = trial["study_sequence"], trial["test_phase"][0]
    slot = _index_by_identity([img for img, _ in study], test["image"])
    target = test["target"]
    target_ids, target_len = word_ids(tokenizer, target)
    studied = {w for _, w in study}
    candidates = [w for w in words if w not in studied]
    random.shuffle(candidates)
    build = lambda seq: pam_eval.build_messages(evaluator, seq, trial["study_prompt"], test["image"], layout="sandwich")
    clean = build(study)
    # Equal token counts in isolation do not guarantee equal counts in the prompt: in the sandwich layout the
    # word's second copy runs into the next pair's "Word:" label, and the tokenizer can merge across that
    # boundary differently for the two words. So render both prompts and keep the first candidate that aligns.
    for alt in candidates:
        alt_ids, alt_len = word_ids(tokenizer, alt)
        if alt_len != target_len or (alt_ids & target_ids):
            continue
        corrupt_study = list(study)
        corrupt_study[slot] = (study[slot][0], alt)
        corrupt = build(corrupt_study)
        if aligned(evaluator, clean, corrupt, prefill=None):
            break
    else:
        raise ValueError(f"no length-matched counterfactual word for {target!r} that keeps the prompts aligned")
    return Pair(
        task="paired_associates", clean=clean, corrupt=corrupt,
        answer=tuple(sorted(target_ids)), counterfactual=tuple(sorted(alt_ids)), slot=slot, n=n,
        meta={"dataset": dataset, "word": target, "counterfactual_word": alt, "wordpool": wordpool,
              **({"item_length": item_length, "token_pool": token_pool} if dataset == token_stimuli.DATASET else {})},
        prefill=None,
    )


BUILDERS = {"afc_recognition": recognition_pair, "serial_order": serial_order_pair, "paired_associates": pam_pair}


def make_pair(evaluator, task, dataset, n, seed, trial, **kw):
    """Seeded like the benchmark: one RNG stream per (seed), advanced per trial."""
    random.seed(f"{seed}:{trial}")
    return BUILDERS[task](evaluator, dataset, n, **kw)
