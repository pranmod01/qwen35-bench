"""Evaluation wrapper for 2-AFC recognition, around tasks/recognition.py.

Study N images, then pick which of 2 images was studied. Scored as a forced choice: the model's
probability of "1" vs "2" after a prefilled `Answer:` turn, not a parse of its free text (the 2B/4B
answer with paragraphs at mid study sizes, which the old regex scored wrongly). The free-text reply
is still generated and stored, to report format compliance.

The `tokens` dataset swaps images for token-ID sequences (tokens/stimuli.py) with the same prompts and
scoring, so image vs text is a config change. It needs `single_foil`, and adds `item_lengths` and
`token_pools` to the condition.
"""
import random
import re

from tokens import stimuli as token_stimuli
from bench.tasks.thinking import thinking_metrics, thinking_record
from src.metrics import calculate_2afc_metrics
from tasks.recognition import AFCRecognitionTask

NAME = "afc_recognition"

# things has no state foils; recombined foils (two studied halves joined) exist only for token items,
# which in turn have no exemplar or state foils.
UNSUPPORTED = {("things", "state"), ("Brady2008", "recombined"), ("things", "recombined"),
               ("tokens", "exemplar"), ("tokens", "state")}

# Novel foils use 2 distinct items per studied image. Brady exemplar/state have 100 pairs, and the
# task silently truncates past that, so the runner refuses instead.
MAX_N_IMAGES = {
    ("Brady2008", "novel"): 1200,
    ("Brady2008", "exemplar"): 100,
    ("Brady2008", "state"): 100,
    ("things", "novel"): 927,
    ("things", "exemplar"): 1854,
}


# With `single_foil` (novel only) an episode draws n + 1 distinct items rather than 2n.
SINGLE_FOIL_MAX_N_IMAGES = {"Brady2008": 2399, "things": 1853}


def max_n_images(condition):
    if condition["dataset"] == token_stimuli.DATASET:
        # Every token is used once per episode; a novel foil takes one more item's worth.
        extra = 1 if condition["foil_type"] == "novel" else 0
        return token_stimuli.max_items(condition["token_pool"], condition["item_length"]) - extra
    if condition.get("single_foil"):
        return SINGLE_FOIL_MAX_N_IMAGES[condition["dataset"]]
    return MAX_N_IMAGES[(condition["dataset"], condition["foil_type"])]


def min_n_images(condition):
    return 3 if condition["foil_type"] == "recombined" else 1


def expand_conditions(task_cfg):
    conditions = []
    for dataset in task_cfg["datasets"]:
        for foil_type in task_cfg["foil_types"]:
            if (dataset, foil_type) in UNSUPPORTED:
                continue
            for n_images in task_cfg["n_images"]:
                for prompt in prompts_from(task_cfg):
                    for stimulus in stimulus_variants(task_cfg, dataset):
                        conditions.append({
                            "dataset": dataset,
                            "foil_type": foil_type,
                            "n_images": n_images,
                            "prompt": prompt,
                            "n_trials": task_cfg.get("n_trials", 10),
                            # Optional, and absent unless set, so earlier result filenames still match.
                            **({"single_foil": True} if task_cfg.get("single_foil") else {}),
                            **stimulus,
                        })
    return conditions


def stimulus_variants(task_cfg, dataset):
    """Token-item settings to cross with each condition; a single empty variant for image datasets."""
    if dataset != token_stimuli.DATASET:
        return [{}]
    # Token episodes always test one item at a random position, which is what single_foil means for
    # images; requiring the flag keeps result filenames honest about it.
    if not task_cfg.get("single_foil"):
        raise ValueError("the tokens dataset needs `single_foil: true`")
    return token_variants(task_cfg)


def token_variants(task_cfg):
    for key in ("item_lengths", "token_pools"):
        if not task_cfg.get(key):
            raise ValueError(f"the tokens dataset needs `{key}`")
    return [{"item_length": length, "token_pool": pool}
            for length in task_cfg["item_lengths"] for pool in task_cfg["token_pools"]]


def parse_response(text):
    """Return 1, 2, or -1 for unparseable."""
    if text is None:
        return -1
    text = text.strip()
    if "1" in text and "2" not in text:
        return 1
    if "2" in text and "1" not in text:
        return 2
    if text.startswith("1"):
        return 1
    if text.startswith("2"):
        return 2
    # Models often answer "The first image..." — no digit in "first"
    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


# Test-turn layouts, chosen per condition by the config's `prompts` list.
# original: vision-memory's wording, question first, then the two unlabeled images.
# labeled:  "Image 1:"/"Image 2:" before each image and the question last. It stopped the 2B/4B
#           answering with paragraphs, but pushed answers toward "1" at small study sizes
#           (2B serial order, n=2: 0.90 -> 0.50).
PROMPTS = {
    "original": (
        "Which of these two images was in the study sequence? "
        "The first image below is 1, the second is 2. "
        "Reply with only the digit 1 or 2 and nothing else."
    ),
    "labeled": (
        "Which of these two images was in the study sequence, Image 1 or Image 2? "
        "Reply with only the digit 1 or 2 and nothing else."
    ),
}


def build_messages(evaluator, study_images, study_prompt, test_images, prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    study_content += token_stimuli.encode_all(evaluator, study_images)
    test_content = build_test_content(evaluator, test_images, prompt)
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": token_stimuli.reword("I have studied the sequence of images.", study_images)},
        {"role": "user", "content": test_content},
    ]


def build_test_content(evaluator, test_images, prompt):
    text = token_stimuli.reword(PROMPTS.get(prompt, ""), test_images)
    if prompt == "original":
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


def run(evaluator, condition, seed):
    n_images = condition["n_images"]
    random.seed(seed)

    tokens = condition["dataset"] == token_stimuli.DATASET
    if tokens:
        pool = token_stimuli.build_pool(evaluator.processor.tokenizer, condition["token_pool"])

    trials = []
    for i in range(condition["n_trials"]):
        if tokens:
            trial_data = token_stimuli.recognition_trials(pool, n_images, condition["item_length"],
                                                          condition["foil_type"])
        else:
            task = AFCRecognitionTask(
                dataset_name=condition["dataset"],
                n_images=n_images,
                foil_type=condition["foil_type"],
                single_foil=condition.get("single_foil", False),
            )
            trial_data = task.get_trials()
        # Without single_foil, test_phase[0] is always the first studied item (test_phase follows study
        # order), so standard-mode results measure list position 1 only. single_foil tests a random position.
        test_trial = trial_data["test_phase"][0]

        messages = build_messages(
            evaluator,
            trial_data["study_sequence"], trial_data["study_prompt"],
            test_trial["images"], condition["prompt"],
        )
        # None with thinking off; otherwise one trace that both the forced and the free answer follow.
        thought = evaluator.think(messages, f"{seed}:{i}")
        forced = evaluator.score_choices(messages, thought=thought)
        response = int(forced["choice"])
        # Kept alongside the forced choice so we can still report how often the model would
        # have answered in the requested format if left to generate freely.
        free = evaluator._call_api(messages, thought=thought)
        trials.append({
            "trial": i,
            "target": test_trial["target"],
            "response": response,
            "correct": 1 if response == test_trial["target"] else 0,
            "foil_type": test_trial["type"],
            "choice_probs": forced["probs"],
            "choice_mass": forced["choice_mass"],
            "top_tokens": forced["top_tokens"],
            "free_response": free,
            "free_parsed": parse_response(free),
            **({"metadata": test_trial["metadata"]} if "metadata" in test_trial else {}),
            **({"thinking": thinking_record(thought)} if thought else {}),
        })

    metrics = calculate_2afc_metrics(trials)
    metrics.update(free_text_metrics(trials))
    metrics.update(thinking_metrics(trials))
    return {"trials": trials, "metrics": metrics}
