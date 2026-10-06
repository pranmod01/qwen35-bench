"""Evaluation wrapper for associative inference, around tasks/associative_inference.py.

Study AB and BC pairs, then pick which of two options is indirectly associated with cue A. The link
A-C is never studied, so the task is inference over two episodic associations rather than recall.

Two pair types: `image` (C is an image, so the test turn holds cue + two option images) and `word`
(C is a word, so the options are text and only the cue is an image). Both are 2-AFC, scored as a
forced choice over "1"/"2" like the other 2-AFC tasks.

Three probes (`probe_types`), so a failure can be located: `inference` (cue A -> C, never studied; the
task itself), `direct_ab` (cue A -> B, studied) and `direct_bc` (cue B -> C, studied; for `word` this is
image->word retrieval, the binding paired associates shows these models can do). A layout is only usable
if both direct probes rise above chance with balanced answers.

Sized in chains, not images: an `image` episode puts 4 images per chain in the study turn (A,B then
B,C) and a `word` episode 3. `n_images` is carried alongside `n_chains` so the dataset/capacity check
and the cross-task x-axis both work in images.
"""
import random

from src.metrics import calculate_associative_inference_metrics
from tasks.associative_inference import AssociativeInferenceTask

NAME = "associative_inference"

IMAGES_PER_CHAIN = {"image": 4, "word": 3}

# Chains a dataset can supply: `image` needs 3 distinct items per chain (A, B, C), `word` needs 2
# (C is a word). Brady2008Objects has 2,400 images, THINGS 1,854 concepts at one exemplar each.
MAX_CHAINS = {
    ("Brady2008", "image"): 800,
    ("Brady2008", "word"): 1200,
    ("things", "image"): 618,
    ("things", "word"): 927,
}

# THINGS builds its dataset with exactly the categories a chain count needs, so a 1-chain episode has
# no spare image for the foil the degenerate case requires and raises. Brady loads all 2,400 anyway.
MIN_CHAINS = {"Brady2008": 1, "things": 2}


def max_n_images(condition):
    key = (condition["dataset"], condition["pair_type"])
    return MAX_CHAINS[key] * IMAGES_PER_CHAIN[condition["pair_type"]]


def min_n_images(condition):
    # Direct probes take their foil from another chain, so they need two.
    chains = MIN_CHAINS[condition["dataset"]] if condition["probe"] == "inference" else 2
    return max(chains, MIN_CHAINS[condition["dataset"]]) * IMAGES_PER_CHAIN[condition["pair_type"]]


def expand_conditions(task_cfg):
    return [
        {
            "dataset": dataset,
            "pair_type": pair_type,
            "probe": probe,
            "n_chains": n_chains,
            # Study images implied by the chain count; what the capacity check and the plots use.
            "n_images": n_chains * IMAGES_PER_CHAIN[pair_type],
            "prompt": prompt,
            "n_trials": task_cfg.get("n_trials", 10),
        }
        for dataset in task_cfg["datasets"]
        for pair_type in task_cfg["pair_types"]
        for probe in task_cfg.get("probe_types", ["inference"])
        for n_chains in task_cfg["n_chains"]
        for prompt in prompts_from(task_cfg)
    ]


# Test-turn layouts. The first sweep used `original` only and produced pure side bias: every model
# answered "2" on 100% of trials, and the direct-association control (control_assoc_inf.py, job
# 12915072) showed even studied A->B retrieval failing the same way, so the turn was not being read.
# Unlike the other 2-AFC tasks this turn carries three images, which is the suspected cause.
#
# original:  "Cue image:" + cue, then the question, then the two options (one turn, three images).
# cue_turn:  the cue is shown and acknowledged in its own turn, so the question turn holds exactly
#            two images -- the shape afc_recognition and serial_order use, where forced choice works.
# labeled:   "Option 1:"/"Option 2:" before each option, question last. Carries the known risk from
#            the 2-AFC prompt A/B (labels pull answers toward "1"), which is why it is measured.
PROMPTS = ("original", "cue_turn", "labeled")
PROBES = ("inference", "direct_ab", "direct_bc")
RELATION = {"inference": "is indirectly associated with", "direct_ab": "was paired with",
            "direct_bc": "was paired with"}


def question(prompt, probe, option_kind):
    """The test question. For `inference` these are the strings the first sweep used, unchanged."""
    relation = RELATION[probe]
    reply = "Reply with only the digit 1 or 2 and nothing else."
    if prompt == "labeled":
        kind = option_kind.capitalize()
        return (f"Which of these two {option_kind}s, {kind} 1 or {kind} 2, {relation} the cue image? "
                + reply)
    order = "The first image below is 1, the second is 2. " if option_kind == "image" else ""
    return f"Which of these two {option_kind}s {relation} the cue image? {order}{reply}"


def prompts_from(task_cfg):
    if not task_cfg.get("prompts"):
        raise ValueError(f"{task_cfg['task']} config needs `prompts`, a list from {sorted(PROMPTS)}")
    unknown_probes = set(task_cfg.get("probe_types", [])) - set(PROBES)
    if unknown_probes:
        raise ValueError(f"unknown probe_types {sorted(unknown_probes)}; expected {list(PROBES)}")
    unknown = set(task_cfg["prompts"]) - set(PROMPTS)
    if unknown:
        raise ValueError(f"unknown prompt(s) {sorted(unknown)}; expected {sorted(PROMPTS)}")
    return task_cfg["prompts"]


def build_messages(evaluator, study_sequence, study_prompt, probe, prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    for pair in study_sequence:
        # vision-memory's label, "(AB pair)" / "(BC pair)": it separates pairs and also says which
        # link of the chain each pair is. The first sweep used a bare "Pair:" here, which dropped the
        # role cue (found in the 2026-10-05 port audit).
        study_content.append({"type": "text", "text": f"({pair['pair_type']} pair)"})
        if "images" in pair:
            study_content += [evaluator._encode_image(img) for img in pair["images"]]
        else:
            study_content.append(evaluator._encode_image(pair["image"]))
            study_content.append({"type": "text", "text": f"Word: {pair['word']}"})

    option_kind = "image" if "images" in probe else "word"
    question_text = {"type": "text", "text": question(prompt, probe["probe"], option_kind)}
    cue = evaluator._encode_image(probe["cue_image"])

    def options(labeled=False):
        if option_kind == "word":
            return [{"type": "text", "text": f"1. {probe['options'][0]}   2. {probe['options'][1]}"}]
        if labeled:
            content = []
            for k, img in enumerate(probe["images"], 1):
                content += [{"type": "text", "text": f"Image {k}:"}, evaluator._encode_image(img)]
            return content
        return [evaluator._encode_image(img) for img in probe["images"]]

    messages = [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the pairs."},
    ]
    if prompt == "original":
        messages.append({"role": "user",
                         "content": [{"type": "text", "text": "Cue image:"}, cue, question_text] + options()})
    elif prompt == "cue_turn":
        # The cue gets its own turn so the question turn holds only the two options.
        messages += [
            {"role": "user", "content": [{"type": "text", "text": "Here is the cue image."}, cue]},
            {"role": "assistant", "content": "I have looked at the cue image."},
            {"role": "user", "content": [question_text] + options()},
        ]
    elif prompt == "labeled":
        messages.append({"role": "user",
                         "content": [{"type": "text", "text": "Cue image:"}, cue]
                                    + options(labeled=True) + [question_text]})
    else:
        raise ValueError(f"unknown prompt {prompt!r}; expected one of {sorted(PROMPTS)}")
    return messages


def swap_options(correct, foil, rng):
    """Randomise side, as the task's own test phase does; returns (options, target)."""
    if rng.random() < 0.5:
        return [correct, foil], 1
    return [foil, correct], 2


def make_probe(trial_data, probe_type, rng):
    """The probe for one episode, on a randomly chosen chain.

    test_phase holds one inference probe per chain in chain order, so test_phase[0] (what the first
    sweep and vision-memory both used) always tested the first-studied chain. Direct probes reuse the
    chosen chain and draw the foil from another chain's same-role item, like the inference foil.
    """
    test_phase = trial_data["test_phase"]
    chain = rng.randrange(len(test_phase))
    base = test_phase[chain]
    if probe_type == "inference":
        return {**base, "probe": "inference"}

    # The task appends every AB pair, then every BC pair, so with n chains AB pair k is
    # study_sequence[k] and BC pair k is study_sequence[n + k].
    n = len(test_phase)
    ab, bc = trial_data["study_sequence"][:n], trial_data["study_sequence"][n:]
    others = [j for j in range(n) if j != chain]
    if not others:
        raise ValueError("direct probes need at least 2 chains for a same-role foil")
    foil = rng.choice(others)
    if probe_type == "direct_ab":
        cue = ab[chain]["images"][0]
        options, target = swap_options(ab[chain]["images"][1], ab[foil]["images"][1], rng)
        key = "images"
    else:
        cue = bc[chain]["images"][0] if "images" in bc[chain] else bc[chain]["image"]
        if "images" in bc[chain]:
            options, target = swap_options(bc[chain]["images"][1], bc[foil]["images"][1], rng)
            key = "images"
        else:
            options, target = swap_options(bc[chain]["word"], bc[foil]["word"], rng)
            key = "options"
    return {"cue_image": cue, key: options, "target": target, "probe": probe_type,
            "metadata": {"chain_index": chain, "foil_chain_index": foil}}


def choice_metrics(trials):
    n = len(trials)
    if not n:
        return {}
    bare = sum((t["free_response"] or "").strip().rstrip(".") in ("1", "2") for t in trials)
    return {
        "free_text_bare_digit": bare / n,
        "mean_p_chosen": sum(max(t["choice_probs"].values()) for t in trials) / n,
        "mean_choice_mass": sum(t["choice_mass"] for t in trials) / n,
        "frac_low_mass": sum(t["choice_mass"] < 0.9 for t in trials) / n,
    }


def run(evaluator, condition, seed):
    n_chains = condition["n_chains"]
    pair_type = condition["pair_type"]
    random.seed(seed)

    # Study images, the cue, and the option images where the options are images.
    n_prompt_images = condition["n_images"] + (3 if pair_type == "image" else 1)
    def episode():
        return AssociativeInferenceTask(
            dataset_name=condition["dataset"],
            n_trials=n_chains * 2,      # the task splits this into AB and BC pairs
            pair_type=pair_type,
        ).get_trials()

    # Built before the probe so the probe uses a real stimulus, then reused as the first trial.
    first_episode = episode()
    first_pair = first_episode["study_sequence"][0]
    sample = first_pair["images"][0] if "images" in first_pair else first_pair["image"]
    if not evaluator.check_image_capacity(n_prompt_images, sample_image=sample):
        return {"skipped": f"model rejected {n_prompt_images} images in one request"}

    trials = []
    for i in range(condition["n_trials"]):
        trial_data = first_episode if i == 0 else episode()
        # Probe draws use their own per-trial stream, so the episodes and the tested chain are the same
        # for every probe type and the inference/direct accuracies compare trial for trial.
        probe = make_probe(trial_data, condition["probe"], random.Random(f"{seed}:{i}"))

        messages = build_messages(
            evaluator, trial_data["study_sequence"], trial_data["study_prompt"],
            probe, condition["prompt"],
        )
        forced = evaluator.score_choices(messages)
        reported = int(forced["choice"])
        free = evaluator._call_api(messages)

        trials.append({
            "trial": i,
            "target": probe["target"],
            "reported": reported,
            "correct": int(reported == probe["target"]),
            "choice_probs": forced["probs"],
            "choice_mass": forced["choice_mass"],
            "top_tokens": forced["top_tokens"],
            "free_response": free,
            "metadata": probe["metadata"],
        })

    metrics = calculate_associative_inference_metrics(trials)
    metrics.update(choice_metrics(trials))
    return {"trials": trials, "metrics": metrics}
