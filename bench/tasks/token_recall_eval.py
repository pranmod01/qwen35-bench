"""Cued recall of token items: the control for afc_recognition's answer step (tokens dataset only).

Study N token items as in recognition, then the answer turn is prefilled with the first half of one studied item and
the model's log-probabilities for the rest are read off (Qwen35Evaluator.score_tail). The 2B fails recognition at n=1 (says "1" on ~95% of trials)
although the studied item is shown verbatim; this separates "the item is not held" from "the item is held but the
1/2 judgment fails". Scored by likelihood, not generation, so there is nothing to parse:

  accuracy           the true tail is more likely than an unstudied tail (summed log-prob); chance 0.5
  first_token_top1   the model's top next token after the cue is the true one; chance ~1 / pool size
  exact_recall       the top token is the true one at every tail position, i.e. greedy decoding would reproduce it

`baseline: true` cues an unstudied item, so accuracy is what the language prior alone gives: ~0.5 for random tokens,
higher for the natural pools, where the true tail is the grammatical continuation of the cue. Memory is the lift over it.
"""
import random

from bench.tasks.afc_recognition_eval import token_variants
from tokens import stimuli as token_stimuli

NAME = "token_recall"

TEST_PROMPT = "Here is the start of one item from the study sequence. Continue it with the rest of that item."


def expand_conditions(task_cfg):
    if task_cfg.get("datasets") != [token_stimuli.DATASET]:
        raise ValueError(f"{NAME} supports only datasets: [{token_stimuli.DATASET}]")
    if any(length < 2 for length in task_cfg.get("item_lengths", [])):
        raise ValueError(f"{NAME} cues with half an item; item_lengths must be >= 2")
    return [
        {"dataset": token_stimuli.DATASET, "n_images": n_images, "n_trials": task_cfg.get("n_trials", 10),
         **({"baseline": True} if task_cfg.get("baseline") else {}), **stimulus}
        for n_images in task_cfg["n_images"]
        for stimulus in token_variants(task_cfg)
    ]


def max_n_images(condition):
    # The unstudied item that supplies the foil tail takes one item's worth of tokens; a baseline's cued item another.
    return token_stimuli.max_items(condition["token_pool"], condition["item_length"]) - 1 - bool(condition.get("baseline"))


def min_n_images(condition):
    return 1


def build_messages(evaluator, trial):
    study_content = [{"type": "text", "text": trial["study_prompt"]}]
    study_content += token_stimuli.encode_all(evaluator, trial["study_sequence"])
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of items."},
        {"role": "user", "content": [{"type": "text", "text": TEST_PROMPT}]},
    ]


def run(evaluator, condition, seed):
    if evaluator.thinking:
        raise ValueError(f"{NAME} does not support thinking: a trace would sit between the cue and the tail")
    random.seed(seed)
    pool = token_stimuli.build_pool(evaluator.processor.tokenizer, condition["token_pool"])

    trials = []
    for i in range(condition["n_trials"]):
        trial = token_stimuli.recall_trials(pool, condition["n_images"], condition["item_length"],
                                            baseline=condition.get("baseline", False))
        # Generation prompt only (ends after the empty think block); the cue is appended as raw IDs.
        prompt = evaluator._prepare_inputs(build_messages(evaluator, trial))
        true = evaluator.score_tail(prompt, trial["cue"], trial["target_tail"])
        foil = evaluator.score_tail(prompt, trial["cue"], trial["foil_tail"])
        correct = sum(true["logprobs"]) > sum(foil["logprobs"])
        trials.append({
            "trial": i,
            "response": "true_tail" if correct else "foil_tail",
            "correct": int(correct),
            "first_token_top1": int(true["top1"][0]),
            "exact_recall": int(all(true["top1"])),
            "true_logprobs": true["logprobs"],
            "foil_logprobs": foil["logprobs"],
            "metadata": {**trial["metadata"], "cue_ids": list(trial["cue"]),
                         "target_tail_ids": list(trial["target_tail"]), "foil_tail_ids": list(trial["foil_tail"])},
        })

    n = len(trials)
    tail_len = len(trials[0]["true_logprobs"])
    metrics = {
        "accuracy": sum(t["correct"] for t in trials) / n,
        "first_token_top1": sum(t["first_token_top1"] for t in trials) / n,
        "exact_recall": sum(t["exact_recall"] for t in trials) / n,
        "mean_true_logprob": sum(sum(t["true_logprobs"]) for t in trials) / (n * tail_len),
        "mean_foil_logprob": sum(sum(t["foil_logprobs"]) for t in trials) / (n * tail_len),
        "n_trials": n,
    }
    return {"trials": trials, "metrics": metrics}
