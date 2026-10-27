"""Evaluation wrapper for paired associates (image-word), around tasks/paired_associate_memory.py.

Study N image-word pairs, then name the word paired with a test image. Message building, parsing and
scoring are ported from vision-memory eval_scripts/eval_pam.py (pair_type='word') so results stay
comparable. The model must produce the word, so chance is near zero rather than 50%.

The `tokens` dataset swaps the images for token-ID sequences (tokens/stimuli.py) as the cue side of each pair;
the word side, the test question and the scoring are unchanged, so image vs token cue is a config change. It
needs `prompts: [sandwich]` (the layout the image runs settled on), and adds `item_lengths` and `token_pools`
to the condition.
"""
import random
import re
from pathlib import Path

from bench.tasks.afc_recognition_eval import token_variants
from bench.tasks.thinking import thinking_metrics, thinking_record
from src import corruption
from src.metrics import calculate_pam_metrics
from tasks.paired_associate_memory import PairedAssociateMemoryTask
from tokens import stimuli as token_stimuli

NAME = "paired_associates"

# imagenet: the concepts in memory_datasets/ImageNet21k_w21_256/concepts.json; update if that list changes.
DATASET_SIZES = {"Brady2008": 2400, "things": 1854, "imagenet": 3808}

# Word pools, chosen per task with `wordpool:` (default `was`, left out of the condition so earlier result
# filenames still match). `was` is the WAS-normed pool from the human PAM literature, which caps a list at
# its 1,638 words; `extended` adds frequent concrete nouns to 4,000 for ImageNet lists longer than that
# (scripts/build_pam_wordpool.py). Each trial samples its words from the whole pool, so the two differ in
# which words can appear, not in how pairs are built.
WORDPOOLS = {"was": "memory_datasets/wasnorm_wordpool.txt",
             "extended": "memory_datasets/pam_wordpool_extended.txt"}
DEFAULT_WORDPOOL = "was"


def load_wordpool(name):
    # The same reading as PairedAssociateMemoryTask._load_wordpool, for the token branch, which has no task object.
    return [line.strip() for line in Path(WORDPOOLS[name]).read_text().splitlines() if line.strip()]


def wordpool_size(name):
    return len(load_wordpool(name))


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
    wordpool = task_cfg.get("wordpool", DEFAULT_WORDPOOL)
    if wordpool not in WORDPOOLS:
        raise ValueError(f"unknown wordpool {wordpool!r}; expected {sorted(WORDPOOLS)}")
    return [
        {"dataset": dataset, "n_images": n_images, "n_trials": task_cfg.get("n_trials", 10),
         **({"prompt": prompt} if prompt else {}),
         **({"wordpool": wordpool} if wordpool != DEFAULT_WORDPOOL else {}),
         **stimulus}
        for dataset in task_cfg["datasets"]
        for n_images in task_cfg["n_images"]
        for prompt in prompts
        for stimulus in stimulus_variants(task_cfg, dataset, prompts)
    ]


def stimulus_variants(task_cfg, dataset, prompts):
    """Token-item settings to cross with each condition; corruption settings (or nothing) for image datasets."""
    if dataset != token_stimuli.DATASET:
        return corruption.variants(task_cfg, dataset)
    # Only the layout the image runs settled on (item 40): the other five exist to explain the image
    # off-by-one, and a token item has no closing boundary token to carry them over to (see build_messages).
    if set(prompts) != {"sandwich"}:
        raise ValueError("the tokens dataset needs `prompts: [sandwich]`")
    return token_variants(task_cfg)


def max_n_images(condition):
    # The task silently takes min(n, dataset, wordpool); the runner refuses instead.
    words = wordpool_size(condition.get("wordpool", DEFAULT_WORDPOOL))
    if condition["dataset"] == token_stimuli.DATASET:
        # Every token is used once per episode; the test cue is a studied item, so no extra item is needed.
        return min(token_stimuli.max_items(condition["token_pool"], condition["item_length"]), words)
    return min(DATASET_SIZES[condition["dataset"]], words)


def min_n_images(condition):
    return 1


# Replaces the task's prompt: with it, the 2B/4B often named the object in the image instead of
# recalling the studied word (PAM at n=1 was only 39-65%).
TEST_PROMPT = (
    "In the study list, which word was shown with this image? Do not name or describe the image. "
    "Reply with only that word and nothing else."
)


def build_messages(evaluator, study_sequence, study_prompt, test_image, test_prompt=TEST_PROMPT,
                   layout="original"):
    tokens = isinstance(test_image, token_stimuli.TokenItem)
    if tokens and layout != "sandwich":
        raise ValueError(f"token items support only the sandwich layout, not {layout!r}")
    # The chat template joins content parts with nothing between them, so with images the text after a pair
    # runs straight into the next pair ("Word: SWORDWord: AXE"); that text is kept as is here. An image is
    # closed by <|vision_end|>, a token item by nothing (tokens/stimuli.ITEM_SEPARATOR opens it), so the text
    # that follows an item starts on a new line, which no item can contain.
    after = "\n" if tokens else ""
    study_content = [{"type": "text", "text": token_stimuli.reword(study_prompt, [test_image])}]
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
        study_content += token_stimuli.encode(evaluator, img)
        if layout == "pair_both":
            study_content.append({"type": "text", "text": f"Word {k}: {word}"})
        elif layout == "tagged":
            study_content.append({"type": "text", "text": f"Pair {k} - Word: {word}"})
        else:
            study_content.append({"type": "text", "text": f"{after}Word: {word}"})
    test_content = token_stimuli.encode(evaluator, test_image)
    test_content.append({"type": "text", "text": after + token_stimuli.reword(test_prompt, [test_image])})
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": token_stimuli.reword("I have studied the image-pairs.", [test_image])},
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

    wordpool = condition.get("wordpool", DEFAULT_WORDPOOL)
    tokens = condition["dataset"] == token_stimuli.DATASET
    if tokens:
        pool = token_stimuli.build_pool(evaluator.processor.tokenizer, condition["token_pool"])
        words = load_wordpool(wordpool)

    trials = []
    for i in range(condition["n_trials"]):
        if tokens:
            trial_data = token_stimuli.pam_trials(pool, n_images, condition["item_length"], words)
        else:
            task = PairedAssociateMemoryTask(dataset_name=condition["dataset"], n_images=n_images, pair_type="word",
                                             wordpool_path=WORDPOOLS[wordpool])
            trial_data = corruption.apply(task.get_trials(), condition, seed)
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
