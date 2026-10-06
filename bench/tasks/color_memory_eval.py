"""Evaluation wrapper for color memory, around tasks/color_memory.py.

Study N objects recolored to known hues, then report the color of one of them shown in grayscale.
Two modes, set per condition:

- named:      the hue is one of six named colors; scored as a forced choice. Chance is 1/6.
- continuous: the hue is any angle on a CIELAB wheel and the model reports degrees; scored by
              circular error. This is the only free-text-scored measure in the benchmark, so its
              parse-failure rate is reported as a first-class number rather than folded into accuracy.

`named` carries the same two answer layouts as MST (`prompts`): color words leak next-token mass to
digits, <|im_end|>, Chinese color words and out-of-set colors like gray and brown
(diag_word_choices.py, jobs 12901178-79), so `numbered` asks for a digit and reshuffles which digit
means which color on every trial.
"""
import random
import re

from src.metrics import calculate_color_metrics, calculate_named_color_metrics
from tasks.color_memory import NAMED_COLORS, ColorMemoryTask

NAME = "color_memory"

COLOR_NAMES = tuple(name for name, _ in NAMED_COLORS)

# memory_datasets/Brady2013ColorObjects holds 540 images; the task silently clamps to that.
MAX_N_IMAGES = 540

MODES = {"named": "named", "continuous": "continuous_color_report"}


def max_n_images(condition):
    return MAX_N_IMAGES


def expand_conditions(task_cfg):
    conditions = []
    for mode in modes_from(task_cfg):
        # The layout choice only exists where there is a closed set of answers to lay out.
        prompts = prompts_from(task_cfg) if mode == "named" else ["degrees"]
        for n_images in task_cfg["n_images"]:
            for prompt in prompts:
                conditions.append({
                    "mode": mode,
                    "n_images": n_images,
                    "prompt": prompt,
                    "n_trials": task_cfg.get("n_trials", 10),
                })
    return conditions


def modes_from(task_cfg):
    if not task_cfg.get("modes"):
        raise ValueError(f"{task_cfg['task']} config needs `modes`, a list from {sorted(MODES)}")
    unknown = set(task_cfg["modes"]) - set(MODES)
    if unknown:
        raise ValueError(f"unknown mode(s) {sorted(unknown)}; expected {sorted(MODES)}")
    return task_cfg["modes"]


STEM = "What color was this item in the study sequence? "

PROMPTS = {
    "words": STEM + f"Choose one: {', '.join(COLOR_NAMES)}. Reply with only the color name and nothing else.",
    # `numbered` and `fixed` share their wording and differ only in whether the digit->color mapping
    # is reshuffled per trial, isolating the cost of holding a per-trial mapping.
    "numbered": STEM + "Reply with only the digit, where {options}.",
    "fixed": STEM + "Reply with only the digit, where {options}.",
    "degrees": (
        "What was the color of this item in the study sequence? "
        "Report the hue angle in degrees on the color wheel. "
        "0/360 degrees is the rightmost point, 90 is the top, "
        "180 is the left, and 270 is the bottom. Use any number from 0 up to 360. "
        "Reply with only the integer angle and nothing else."
    ),
}


def prompts_from(task_cfg):
    layouts = {"words", "numbered", "fixed"}
    if not task_cfg.get("prompts"):
        raise ValueError(f"{task_cfg['task']} config needs `prompts`, a list from {sorted(layouts)}")
    unknown = set(task_cfg["prompts"]) - layouts
    if unknown:
        raise ValueError(f"unknown prompt(s) {sorted(unknown)}; expected {sorted(layouts)}")
    return task_cfg["prompts"]


def answer_layout(prompt):
    """One trial's prompt text, choice tokens, and the map from choice token back to a color name."""
    if prompt == "words":
        return PROMPTS["words"], COLOR_NAMES, {c: c for c in COLOR_NAMES}
    if prompt in ("numbered", "fixed"):
        order = list(COLOR_NAMES)
        if prompt == "numbered":
            random.shuffle(order)
        answer_map = {str(i): color for i, color in enumerate(order, 1)}
        options = ", ".join(f"{digit} = {color}" for digit, color in answer_map.items())
        return PROMPTS[prompt].format(options=options), tuple(answer_map), answer_map
    raise ValueError(f"unknown prompt {prompt!r}; expected 'words' or 'numbered'")


def build_messages(evaluator, study_sequence, study_prompt, probe, test_prompt, color_wheel=None):
    study_content = [{"type": "text", "text": study_prompt}]
    study_content += [evaluator._encode_image(img) for img in study_sequence]
    test_content = [evaluator._encode_image(probe)]
    if color_wheel is not None:
        # The wheel is the response scale, so it has to be in the test turn and labelled as itself,
        # or it reads as one more object to remember.
        test_content += [{"type": "text", "text": "Color wheel:"}, evaluator._encode_image(color_wheel)]
    test_content.append({"type": "text", "text": test_prompt})
    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the colors."},
        {"role": "user", "content": test_content},
    ]


def parse_degrees(text):
    """Last number in [0, 360] in the reply, wrapped into [0, 360); -1 if there is none.

    vision-memory's parse_angle_response, unchanged in behaviour. The first port took the *first*
    1-3 digit run, which read "-30" as 30 and "1234" as 123, and picked up numbers a reply mentions
    before its answer (fixed 2026-10-05, port audit).
    """
    if text is None:
        return -1
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    candidates = [float(m.group()) for m in re.finditer(r"-?\d+(?:\.\d+)?", text)]
    candidates = [v for v in candidates if 0 <= v <= 360]
    if not candidates:
        return -1
    return candidates[-1] % 360


def choice_metrics(trials):
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
    mode = condition["mode"]
    random.seed(seed)

    # Continuous mode also puts the color wheel in the test turn.
    n_prompt_images = n_images + (2 if mode == "continuous" else 1)
    # Built before the probe so the probe uses a real stimulus, then reused as the first trial.
    first_episode = ColorMemoryTask(n_images=n_images, mode=MODES[mode]).get_trials()
    if not evaluator.check_image_capacity(n_prompt_images, sample_image=first_episode["study_sequence"][0]):
        return {"skipped": f"model rejected {n_prompt_images} images in one request"}

    trials = []
    for i in range(condition["n_trials"]):
        trial_data = (first_episode if i == 0
                      else ColorMemoryTask(n_images=n_images, mode=MODES[mode]).get_trials())
        probe = trial_data["test_phase"][0]

        if mode == "named":
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
                "target": probe["target"],
                "reported": reported,
                "correct": int(reported == probe["target"]),
                "choice": forced["choice"],
                "answer_map": answer_map,
                "choice_probs": forced["probs"],
                "choice_mass": forced["choice_mass"],
                "top_tokens": forced["top_tokens"],
                "free_response": free,
                "metadata": probe["metadata"],
            })
        else:
            messages = build_messages(
                evaluator, trial_data["study_sequence"], trial_data["study_prompt"],
                probe["image"], PROMPTS["degrees"], color_wheel=probe["color_wheel"],
            )
            raw = evaluator._call_api(messages)
            reported = parse_degrees(raw)
            error = None if reported == -1 else abs((reported - probe["target"] + 180) % 360 - 180)
            trials.append({
                "trial": i,
                "target": probe["target"],
                "reported": reported,
                # No per-trial "correct" for a continuous report; the cell's accuracy comes from the
                # circular error below. Within 30 degrees is recorded so a rate can still be read off.
                "correct": None if error is None else int(error <= 30),
                "abs_error_degrees": error,
                "raw_response": raw,
                "metadata": probe["metadata"],
            })

    if mode == "named":
        metrics = calculate_named_color_metrics([t["reported"] for t in trials],
                                                [t["target"] for t in trials])
        metrics.update(choice_metrics(trials))
    else:
        scored = [t for t in trials if t["reported"] != -1]
        metrics = calculate_color_metrics([t["reported"] for t in scored],
                                          [t["target"] for t in scored]) if scored else {"accuracy": 0}
        metrics["parse_failure_rate"] = sum(t["reported"] == -1 for t in trials) / len(trials)
        metrics["frac_within_30_degrees"] = sum(bool(t["correct"]) for t in trials) / len(trials)
    return {"trials": trials, "metrics": metrics}
