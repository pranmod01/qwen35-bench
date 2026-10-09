"""Evaluation wrapper for paired associates (image-word), around tasks/paired_associate_memory.py.

Study N image-word pairs, then name the word paired with a test image. Message building, parsing and
scoring are ported from vision-memory eval_scripts/eval_pam.py (pair_type='word') so results stay
comparable. The model must produce the word, so chance is near zero rather than 50%.
"""
import random
import re

from bench.tasks.thinking import thinking_metrics, thinking_record
from src.metrics import calculate_pam_metrics
from tasks.paired_associate_memory import PairedAssociateMemoryTask

NAME = "paired_associates"

WORDPOOL_SIZE = 1637          # memory_datasets/wasnorm_wordpool.txt
DATASET_SIZES = {"Brady2008": 2400, "things": 1854}


# Study-list layouts. `original` has nothing between pairs, so the text right before image k is the
# previous pair's word ("Word: W_{k-1}"), and a model that binds an image to the text beside it recalls
# exactly that word. `pair_numbered` opens every pair with "Pair k:" so each image is preceded by its
# own marker instead.
#
# `pair_numbered` fixed that off-by-one for the 27B (THINGS n=500: 54 previous-pair swaps -> 2) and
# *caused* it for the 9B (6 -> 35, accuracy 0.74 -> 0.26). Its marker changes two things at once: it
# inserts a boundary, and it inserts a running index that reaches 500. `pair_plain` is the control that
# separates them -- the same boundary with no number -- so whichever effect is responsible shows up as
# the difference between the two.
# `pair_plain` (2026-09-30) settled which half of `pair_numbered` does the work: dropping the index and
# keeping only the boundary made the off-by-one far worse for both models (27B THINGS n=100: 1 previous-
# pair swap -> 55). The index is the identifier, not the separator. `pair_both` and `tagged` follow that
# up by repeating the index on the *word* as well, so the binding is stated on both sides rather than
# left to adjacency; `tagged` also names which element is which.
#
#   original       [img] Word: SWORD
#   pair_numbered  Pair 7: [img] Word: SWORD
#   pair_plain     Pair: [img] Word: SWORD
#   pair_both      Pair 7: [img] Word 7: SWORD
#   tagged         Pair 7 - Image: [img] Pair 7 - Word: SWORD
#   sandwich       Word: SWORD [img] Word: SWORD
#
# `pair_both` and `tagged` (2026-10-04) removed the off-by-one for the 9B (0 previous-pair swaps in all
# four cells) and made it worse for the 27B (Brady n=500: 51 and 42 of 100). Every error either model
# makes is the word *before* the image, so `sandwich` puts the image's own word on both sides: the text
# right before each image is then its own word, whichever side a model binds to.
PROMPTS = {"original", "pair_numbered", "pair_plain", "pair_both", "tagged", "sandwich"}


def expand_conditions(task_cfg):
    # `prompts` is optional here, unlike the other tasks: PAM results predate it, and absent means
    # `original` with the key left out of the condition, so those result filenames still match.
    # Listing `original` explicitly gives new filenames and reruns it.
    prompts = task_cfg.get("prompts", [None])
    unknown = set(prompts) - PROMPTS - {None}
    if unknown:
        raise ValueError(f"unknown prompt(s) {sorted(unknown)}; expected {sorted(PROMPTS)}")
    return [
        {"dataset": dataset, "n_images": n_images, "n_trials": task_cfg.get("n_trials", 10),
         **({"prompt": prompt} if prompt else {})}
        for dataset in task_cfg["datasets"]
        for n_images in task_cfg["n_images"]
        for prompt in prompts
    ]


def max_n_images(condition):
    # The task silently takes min(n, dataset, wordpool); the runner refuses instead.
    return min(DATASET_SIZES[condition["dataset"]], WORDPOOL_SIZE)


# Replaces the task's prompt: with it, the 2B/4B often named the object in the image instead of
# recalling the studied word (PAM at n=1 was only 39-65%).
TEST_PROMPT = (
    "In the study list, which word was shown with this image? Do not name or describe the image. "
    "Reply with only that word and nothing else."
)


def build_messages(evaluator, study_sequence, study_prompt, test_image, test_prompt=TEST_PROMPT,
                   layout="original"):
    study_content = [{"type": "text", "text": study_prompt}]
    for k, (img, word) in enumerate(study_sequence, 1):
        if layout == "pair_numbered":
            study_content.append({"type": "text", "text": f"Pair {k}:"})
        elif layout == "pair_plain":
            study_content.append({"type": "text", "text": "Pair:"})
        elif layout == "pair_both":
            study_content.append({"type": "text", "text": f"Pair {k}:"})
        elif layout == "tagged":
            study_content.append({"type": "text", "text": f"Pair {k} - Image:"})
        elif layout == "sandwich":
            study_content.append({"type": "text", "text": f"Word: {word}"})
        study_content.append(evaluator._encode_image(img))
        if layout == "pair_both":
            study_content.append({"type": "text", "text": f"Word {k}: {word}"})
        elif layout == "tagged":
            study_content.append({"type": "text", "text": f"Pair {k} - Word: {word}"})
        else:
            study_content.append({"type": "text", "text": f"Word: {word}"})
    test_content = [evaluator._encode_image(test_image), {"type": "text", "text": test_prompt}]
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the image-pairs."},
        {"role": "user", "content": test_content},
    ]


def parse_word_response(text, target_word):
    """Extract the reported word, lowercased."""
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    # Models echo the study format ("Word: ANTLER"); without this the label itself was parsed as the answer.
    # Accuracy was unaffected (the target check below runs first), only the reported word.
    text = re.sub(r"^\s*word\s*:\s*", "", text, flags=re.IGNORECASE)
    words = text.split()

    if len(words) == 1:
        return words[0].lower().strip('.,;:!?"\'')
    if target_word.lower() in text.lower():
        return target_word.lower()
    # Wordpool words are ALL CAPS, so prefer the first all-caps token
    for word in words:
        cleaned = word.strip('.,;:!?"\'')
        if cleaned and cleaned.isupper() and len(cleaned) > 1:
            return cleaned.lower()
    for word in reversed(words):
        cleaned = word.strip('.,;:!?"\'')
        if cleaned:
            return cleaned.lower()
    return text.lower()


def run(evaluator, condition, seed):
    n_images = condition["n_images"]
    random.seed(seed)

    trials = []
    for i in range(condition["n_trials"]):
        task = PairedAssociateMemoryTask(dataset_name=condition["dataset"], n_images=n_images, pair_type="word")
        trial_data = task.get_trials()
        test_trial = trial_data["test_phase"][0]

        messages = build_messages(
            evaluator,
            trial_data["study_sequence"], trial_data["study_prompt"],
            test_trial["image"],
            layout=condition.get("prompt") or "original",
        )
        thought = evaluator.think(messages, f"{seed}:{i}")
        raw = evaluator._call_api(messages, thought=thought)
        reported = parse_word_response(raw, test_trial["target"])
        study_words = [w.lower() for _, w in trial_data["study_sequence"]]
        study_position = study_words.index(test_trial["target"].lower()) + 1
        reported_position = study_words.index(reported.lower()) + 1 if reported.lower() in study_words else None
        trials.append({
            "study_position": study_position,  # 1-based; for recency/primacy analysis
            "trial": i,
            "target": test_trial["target"],
            "reported": reported,
            "correct": int(reported.lower() == test_trial["target"].lower()),
            "raw_response": raw,
            "metadata": test_trial["metadata"],
            # A wrong answer that was still a studied word means a swapped pair, not a made-up word
            "reported_in_study_list": reported in study_words,
            # Where in the list a swapped word came from; -1 is the pair studied just before the target.
            # Recorded so a layout-driven swap shows up in every run without regenerating the lists.
            "swap_offset": (reported_position - study_position
                            if reported_position and reported_position != study_position else None),
            **({"thinking": thinking_record(thought)} if thought else {}),
        })

    metrics = calculate_pam_metrics(trials)
    metrics["frac_reported_in_study_list"] = sum(t["reported_in_study_list"] for t in trials) / len(trials)
    offsets = [t["swap_offset"] for t in trials if t["swap_offset"] is not None]
    metrics["n_swaps"] = len(offsets)
    metrics["frac_swaps_prev_pair"] = sum(o == -1 for o in offsets) / len(offsets) if offsets else None
    metrics.update(thinking_metrics(trials))
    return {"trials": trials, "metrics": metrics}
