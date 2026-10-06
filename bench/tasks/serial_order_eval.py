"""Evaluation wrapper for serial order (2-AFC), around tasks/serial_order_memory.py.

Study N images, then say which of two studied images came first. Scored as a forced choice on the
model's probability of "1" vs "2" (see afc_recognition_eval); the free-text reply is stored too.
Chance is 50%.

The `tokens` dataset swaps images for token-ID sequences (tokens/stimuli.py) with the same prompts and
scoring; it adds `item_lengths` and `token_pools` to the condition.
"""
import math
import random
import re

from bench.tasks.afc_recognition_eval import token_variants
from tokens import stimuli as token_stimuli
from src.metrics import calculate_afc_serial_order_metrics
from tasks.serial_order_memory import AFCSerialOrderMemoryTask

NAME = "serial_order"

DATASET_SIZES = {"Brady2008": 2400, "things": 1854}


def expand_conditions(task_cfg):
    # `exemplars_per_concept` and `counterbalance` are optional and left out of the condition at their
    # defaults, so earlier result filenames still match. Only THINGS supports more than one photo per
    # concept.
    k = task_cfg.get("exemplars_per_concept", 1)
    counterbalance = bool(task_cfg.get("counterbalance", False))
    return [
        {"dataset": dataset, "n_images": n_images, "prompt": prompt, "n_trials": task_cfg.get("n_trials", 10),
         **({"exemplars_per_concept": k} if k > 1 else {}),
         **({"counterbalance": True} if counterbalance else {}),
         **stimulus}
        for dataset in task_cfg["datasets"]
        for n_images in task_cfg["n_images"]
        for prompt in prompts_from(task_cfg)
        for stimulus in (token_variants(task_cfg) if dataset == token_stimuli.DATASET else [{}])
    ]


def max_n_images(condition):
    if condition["dataset"] == token_stimuli.DATASET:
        return token_stimuli.max_items(condition["token_pool"], condition["item_length"])
    return DATASET_SIZES[condition["dataset"]] * condition.get("exemplars_per_concept", 1)


def min_n_images(condition):
    return 2  # need two studied images to compare


# Test-turn layouts, chosen per condition by the config's `prompts` list; see
# afc_recognition_eval.PROMPTS. Under "labeled", "appeared first ... Image 1" likely pairs "first"
# with "1": the 2B answered "1" on 100% of n=2 trials.
PROMPTS = {
    "original": "Which of these two images appeared first in the study sequence? Reply with only the digit 1 or 2 and nothing else.",
    "labeled": (
        "Which of these two images appeared first in the study sequence, Image 1 or Image 2? "
        "Reply with only the digit 1 or 2 and nothing else."
    ),
    # Same layout as `original`, but asks for the later image, so the correct digit flips. Tests whether
    # the 9B's "1" preference is for the digit itself or comes from pairing "first" with the image
    # shown first (2026-09-28, configs/serial_order_later_ab.yaml).
    "later": "Which of these two images appeared later in the study sequence? Reply with only the digit 1 or 2 and nothing else.",
}


def build_messages(evaluator, study_sequence, study_prompt, test_images, prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    study_content += token_stimuli.encode_all(evaluator, study_sequence)
    test_content = build_test_content(evaluator, test_images, prompt)
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": token_stimuli.reword("I have studied the sequence of images.", study_sequence)},
        {"role": "user", "content": test_content},
    ]


def parse_afc_response(text):
    """First standalone 1 or 2, since models tend to lead with the answer."""
    if text is None:
        return -1
    text = text.strip()
    m = re.search(r"(?<!\d)[12](?!\d)", text)
    if m:
        return int(m.group())
    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


def build_test_content(evaluator, test_images, prompt):
    text = token_stimuli.reword(PROMPTS.get(prompt, ""), test_images)
    if prompt in ("original", "later"):
        return [{"type": "text", "text": text}] + token_stimuli.encode_all(evaluator, test_images)
    if prompt == "labeled":
        content = []
        for k, img in enumerate(test_images, 1):
            content += [{"type": "text", "text": token_stimuli.reword(f"Image {k}:", test_images)}]
            content += token_stimuli.encode(evaluator, img)
        return content + [{"type": "text", "text": text}]
    raise ValueError(f"unknown prompt {prompt!r}; expected one of {sorted(PROMPTS)}")


def prompts_from(task_cfg):
    # Required rather than defaulted: the two layouts give different answers, so which one produced
    # a result must be explicit in the config and the result filename.
    if not task_cfg.get("prompts"):
        raise ValueError(f"{task_cfg['task']} config needs `prompts`, a list from {sorted(PROMPTS)}")
    unknown = set(task_cfg["prompts"]) - set(PROMPTS)
    if unknown:
        raise ValueError(f"unknown prompt(s) {sorted(unknown)}; expected {sorted(PROMPTS)}")
    return task_cfg["prompts"]


def free_text_metrics(trials):
    """How the free-text reply compares to the forced choice, i.e. the cost of the old scoring."""
    n = len(trials)
    if not n:
        return {}
    chosen = [t.get("response", t.get("reported")) for t in trials]
    bare = sum((t["free_response"] or "").strip().rstrip(".") in ("1", "2") for t in trials)
    parsed = [(t["free_parsed"], c) for t, c in zip(trials, chosen) if t["free_parsed"] != -1]
    return {
        "free_text_bare_digit": bare / n,
        "free_text_unparseable": sum(t["free_parsed"] == -1 for t in trials) / n,
        "free_text_agreement": (sum(p == c for p, c in parsed) / len(parsed)) if parsed else None,
        "mean_p_chosen": sum(max(t["choice_probs"].values()) for t in trials) / n,
        "mean_choice_mass": sum(t["choice_mass"] for t in trials) / n,
        "frac_low_mass": sum(t["choice_mass"] < 0.9 for t in trials) / n,
    }


def log_odds(probs, digit):
    # Probs are renormalised over "1"/"2", so they can round to exactly 0 or 1 in float32.
    p = min(max(probs[str(digit)], 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def counterbalance_metrics(trials):
    n = len(trials)
    return {
        "accuracy_order_shown": sum(t["correct_order_shown"] for t in trials) / n,
        "accuracy_order_swapped": sum(t["correct_order_swapped"] for t in trials) / n,
        # Same image chosen under both orders. A model answering by test position alone scores 0.
        "frac_consistent": sum(t["consistent"] for t in trials) / n,
        "n_ties": sum(t["log_odds_target"] == 0 for t in trials),
    }


def run(evaluator, condition, seed):
    n_images = condition["n_images"]
    random.seed(seed)

    tokens = condition["dataset"] == token_stimuli.DATASET
    if tokens:
        pool = token_stimuli.build_pool(evaluator.processor.tokenizer, condition["token_pool"])

    trials = []
    for i in range(condition["n_trials"]):
        if tokens:
            trial_data = token_stimuli.serial_order_trials(pool, n_images, condition["item_length"])
        else:
            task = AFCSerialOrderMemoryTask(dataset_name=condition["dataset"], n_images=n_images, n_tests=1,
                                            exemplars_per_concept=condition.get("exemplars_per_concept", 1))
            trial_data = task.get_trials()
        test_trial = trial_data["test_phase"][0]
        earlier = test_trial["target"]
        target = 3 - earlier if condition["prompt"] == "later" else earlier

        messages = build_messages(
            evaluator,
            trial_data["study_sequence"], trial_data["study_prompt"],
            test_trial["images"], condition["prompt"],
        )
        forced = evaluator.score_choices(messages)
        reported = int(forced["choice"])
        # Kept alongside the forced choice so we can still report how often the model would
        # have answered in the requested format if left to generate freely.
        free = evaluator._call_api(messages)
        trial = {
            "trial": i,
            "target": target,
            "earlier_image": earlier,  # which test image was studied first, whatever the question asked
            "reported": reported,
            "correct": int(reported == target),
            "choice_probs": forced["probs"],
            "choice_mass": forced["choice_mass"],
            "top_tokens": forced["top_tokens"],
            "free_response": free,
            "free_parsed": parse_afc_response(free),
            "metadata": test_trial["metadata"],
        }
        if condition.get("counterbalance"):
            trial.update(score_swapped(evaluator, trial_data, test_trial, condition["prompt"], target, forced))
        trials.append(trial)

    metrics = calculate_afc_serial_order_metrics(trials)
    metrics.update(free_text_metrics(trials))
    if condition.get("counterbalance"):
        metrics.update(counterbalance_metrics(trials))
    return {"trials": trials, "metrics": metrics}


def score_swapped(evaluator, trial_data, test_trial, prompt, target, forced):
    """Score the same trial with the two test images swapped, and combine the two orders.

    A preference for answering "1" (or for the image shown first in the test) adds the same amount
    to both orders' log-odds for that position, so summing the log-odds for the *same image* across
    the two orders cancels it; what is left is evidence for that image. The combined choice replaces
    `reported`/`correct`, still labelled by its position in the order shown, so hit1/hit2 and the
    BIAS flag keep their meaning. Ties (log-odds exactly 0) resolve to the order shown's choice.
    No random draws happen here, so episodes stay identical to the uncounterbalanced runs.
    """
    swapped_messages = build_messages(
        evaluator,
        trial_data["study_sequence"], trial_data["study_prompt"],
        test_trial["images"][::-1], prompt,
    )
    swapped = evaluator.score_choices(swapped_messages)
    lo_target = log_odds(forced["probs"], target) + log_odds(swapped["probs"], 3 - target)
    reported_shown = int(forced["choice"])
    reported_swapped = int(swapped["choice"])
    if lo_target > 0:
        reported = target
    elif lo_target < 0:
        reported = 3 - target
    else:
        reported = reported_shown
    return {
        "reported": reported,
        "correct": int(reported == target),
        "log_odds_target": lo_target,
        "reported_order_shown": reported_shown,
        "correct_order_shown": int(reported_shown == target),
        "choice_probs_swapped": swapped["probs"],
        "choice_mass_swapped": swapped["choice_mass"],
        "reported_order_swapped": reported_swapped,
        "correct_order_swapped": int(reported_swapped == 3 - target),
        # Same image picked in both orders: in the swapped order that image sits at 3 - position.
        "consistent": int(reported_swapped == 3 - reported_shown),
    }
