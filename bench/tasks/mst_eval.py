"""Evaluation wrapper for the Mnemonic Similarity Task, around tasks/mnemonic_similarity.py.

Study N target images, then judge one probe: 'old' (exact repeat), 'similar' (a lure, i.e. a different
version of a studied object) or 'new' (never studied). One probe per episode, drawn from the shuffled
test phase, so target/lure/foil land in roughly equal thirds across a cell.

Scored as a forced choice, with two answer layouts under test (`prompts`). Words and digits are not
interchangeable here: asked for a word, the models put only 0.07-0.86 of their next-token mass on
old/similar/new, leaking it to digits (the 4B answers "1" at up to 0.64), <|im_end|>, Chinese
translations and capitalised variants (diag_word_choices.py, jobs 12901178-79). `numbered` asks for a
digit and reshuffles which digit means which answer on every trial, so a digit preference shows up as
chance accuracy instead of as memory.
"""
import random

from src.metrics import calculate_mst_metrics
from tasks.mnemonic_similarity import MnemonicSimilarityTask

NAME = "mst"

ANSWERS = ("old", "similar", "new")

# 6 sets x 192 pairs = 1,152, and an episode needs as many unstudied pairs as studied ones for foils.
# The task silently clamps to len(pairs)//2; the runner refuses instead.
MAX_N_IMAGES = 576


def max_n_images(condition):
    return MAX_N_IMAGES


def expand_conditions(task_cfg):
    return [
        {"n_images": n_images, "prompt": prompt, "n_trials": task_cfg.get("n_trials", 10)}
        for n_images in task_cfg["n_images"]
        for prompt in prompts_from(task_cfg)
    ]


# The stem is identical across layouts so the A/B isolates the answer format. It is the wording from
# tasks/mnemonic_similarity.py, which the wrapper now owns (as for the other ported tasks).
STEM = (
    "Some test images are exact matches to study images, some show a "
    "different version of an object you studied (same category but a "
    "different photo, angle, or exemplar - these are NOT identical to "
    "any study image), and some are completely new objects you never saw. "
    "Is this test image 'old' (exact match to a study image), "
    "'similar' (different version of a studied object), "
    "or 'new' (object you never saw)? "
)

# `numbered` and `fixed` share their wording and differ only in whether the digit->answer mapping is
# reshuffled per trial, so the pair isolates the cost of holding a per-trial mapping.
NUMBERED = STEM + "Reply with only the digit, where {options}."

PROMPTS = {
    "words": STEM + "Reply with one word: old, similar, or new.",
    "numbered": NUMBERED,
    "fixed": NUMBERED,
}


def prompts_from(task_cfg):
    # Required rather than defaulted: the layouts are scored over different token sets, so which one
    # produced a result has to be explicit in the config and in the result filename.
    if not task_cfg.get("prompts"):
        raise ValueError(f"{task_cfg['task']} config needs `prompts`, a list from {sorted(PROMPTS)}")
    unknown = set(task_cfg["prompts"]) - set(PROMPTS)
    if unknown:
        raise ValueError(f"unknown prompt(s) {sorted(unknown)}; expected {sorted(PROMPTS)}")
    return task_cfg["prompts"]


def answer_layout(prompt):
    """One trial's prompt text, choice tokens, and the map from choice token back to an answer."""
    if prompt == "words":
        return PROMPTS["words"], ANSWERS, {a: a for a in ANSWERS}
    if prompt in ("numbered", "fixed"):
        order = list(ANSWERS)
        if prompt == "numbered":
            random.shuffle(order)
        answer_map = {str(i): answer for i, answer in enumerate(order, 1)}
        options = ", ".join(f"{digit} = {answer}" for digit, answer in answer_map.items())
        return PROMPTS[prompt].format(options=options), tuple(answer_map), answer_map
    raise ValueError(f"unknown prompt {prompt!r}; expected one of {sorted(PROMPTS)}")


def build_messages(evaluator, study_sequence, study_prompt, probe_image, test_prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    study_content += [evaluator._encode_image(img) for img in study_sequence]
    # Image before text, as in pam_eval: with the question first the models answered about the study
    # sequence instead of the probe.
    test_content = [evaluator._encode_image(probe_image), {"type": "text", "text": test_prompt}]
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content},
    ]


def choice_metrics(trials):
    """Whether the forced choice was a real choice: how much mass sat on the answer tokens at all."""
    n = len(trials)
    if not n:
        return {}
    free_matches = sum((t["free_response"] or "").strip().strip(".").lower() == t["reported"] for t in trials)
    return {
        "mean_p_chosen": sum(max(t["choice_probs"].values()) for t in trials) / n,
        "mean_choice_mass": sum(t["choice_mass"] for t in trials) / n,
        "frac_low_mass": sum(t["choice_mass"] < 0.9 for t in trials) / n,
        "free_text_agreement": free_matches / n,
    }


def run(evaluator, condition, seed):
    n_images = condition["n_images"]
    random.seed(seed)

    # Built before the probe so the probe uses a real stimulus, then reused as the first trial.
    first_episode = MnemonicSimilarityTask(n_study=n_images).get_trials()
    if not evaluator.check_image_capacity(n_images + 1, sample_image=first_episode["study_sequence"][0]):
        return {"skipped": f"model rejected {n_images + 1} images in one request"}

    trials = []
    for i in range(condition["n_trials"]):
        trial_data = first_episode if i == 0 else MnemonicSimilarityTask(n_study=n_images).get_trials()
        # The test phase is shuffled and holds n targets, n lures and n foils; one probe per episode
        # keeps episodes independent, as in the other tasks.
        probe = trial_data["test_phase"][0]

        test_prompt, choices, answer_map = answer_layout(condition["prompt"])
        messages = build_messages(
            evaluator, trial_data["study_sequence"], trial_data["study_prompt"],
            probe["image"], test_prompt,
        )
        forced = evaluator.score_choices(messages, choices=choices)
        reported = answer_map[forced["choice"]]
        free = evaluator._call_api(messages)

        trials.append({
            "trial": i,
            "type": probe["type"],                    # target | lure | foil, read by calculate_mst_metrics
            "target": probe["target"],
            "reported": reported,
            "correct": int(reported == probe["target"]),
            "choice": forced["choice"],
            "answer_map": answer_map,                 # which digit meant what on this trial
            "choice_probs": forced["probs"],
            "choice_mass": forced["choice_mass"],
            "top_tokens": forced["top_tokens"],
            "free_response": free,
            "metadata": probe["metadata"],            # carries the lure's similarity bin, for LDI
        })

    metrics = calculate_mst_metrics(trials)
    metrics.update(choice_metrics(trials))
    return {"trials": trials, "metrics": metrics}
