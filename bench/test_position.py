"""Checks for the pinned test position in recognition (`test_positions`). No model or tokenizer needed; the image
check reads a few Brady objects from memory_datasets. Run from the repo root with the venv python:
python -m bench.test_position"""
import random

from bench.tasks import afc_recognition_eval as afc
from tasks.recognition import AFCRecognitionTask
from tokens import stimuli as token_stimuli

TASK_CFG = {"task": "afc_recognition", "prompts": ["original"], "datasets": ["Brady2008"], "foil_types": ["novel"],
            "single_foil": True, "n_images": [5, 100], "n_trials": 50}


def refused(call):
    try:
        call()
    except ValueError:
        return True
    return False


def test_conditions():
    plain = afc.expand_conditions(TASK_CFG)
    assert all("test_position" not in c for c in plain)   # filenames of random-position results unchanged
    pinned = afc.expand_conditions({**TASK_CFG, "test_positions": [0, 0.5, 1]})
    assert len(pinned) == 3 * len(plain)
    index = {(c["n_images"], c["test_position"]): afc.position_index(c) for c in pinned}
    assert index == {(5, 0): 1, (5, 0.5): 3, (5, 1): 5, (100, 0): 1, (100, 0.5): 51, (100, 1): 100}, index
    assert afc.position_index(plain[0]) is None
    assert refused(lambda: afc.expand_conditions({**TASK_CFG, "test_positions": [0.5], "single_foil": False}))
    assert refused(lambda: afc.expand_conditions({**TASK_CFG, "test_positions": [1.5]}))


def test_token_trials():
    pool = list(range(1000))
    for position in (1, 4, 10):
        for seed in range(20):
            random.seed(seed)
            free = token_stimuli.recognition_trials(pool, 10, 8, "novel")
            random.seed(seed)
            pinned = token_stimuli.recognition_trials(pool, 10, 8, "novel", test_position=position)
            test, free_test = pinned["test_phase"][0], free["test_phase"][0]
            assert test["metadata"]["test_position"] == position
            assert pinned["study_sequence"][position - 1] == test["images"][test["target"] - 1]
            # Same episode as the unpinned draw: same options in the same order, same studied items.
            assert test["images"] == free_test["images"] and test["target"] == free_test["target"]
            assert sorted(i.ids for i in pinned["study_sequence"]) == sorted(i.ids for i in free["study_sequence"])
    assert refused(lambda: token_stimuli.recognition_trials(pool, 10, 8, "recombined", test_position=2))
    assert refused(lambda: token_stimuli.recognition_trials(pool, 10, 8, "novel", test_position=11))


def test_image_trials():
    def episode(seed, position):
        random.seed(seed)
        trial = AFCRecognitionTask(dataset_name="Brady2008", n_images=6, foil_type="novel", single_foil=True,
                                   test_position=position).get_trials()
        (test,) = trial["test_phase"]
        study = [img.tobytes() for img in trial["study_sequence"]]
        return study, [img.tobytes() for img in test["images"]], test

    seen = set()
    for seed in range(8):
        free_study, free_options, free_test = episode(seed, None)
        # Unpinned trials record where the shuffle put the tested item.
        assert free_study[free_test["metadata"]["test_position"] - 1] == free_options[free_test["target"] - 1]
        seen.add(free_test["metadata"]["test_position"])
        for position in (1, 3, 6):
            study, options, test = episode(seed, position)
            assert test["metadata"]["test_position"] == position
            assert study[position - 1] == options[test["target"] - 1]
            assert options == free_options and test["target"] == free_test["target"]
            assert sorted(study) == sorted(free_study)
    assert len(seen) > 1, seen
    assert refused(lambda: AFCRecognitionTask(n_images=6, foil_type="novel", test_position=2))
    assert refused(lambda: AFCRecognitionTask(n_images=6, foil_type="novel", single_foil=True, test_position=7))


if __name__ == "__main__":
    tests = [test_conditions, test_token_trials, test_image_trials]
    for test in tests:
        test()
    print(f"all {len(tests)} checks passed")
