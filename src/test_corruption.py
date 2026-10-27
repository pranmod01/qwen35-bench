"""Checks for the image corruptions and their config axis. CPU only; needs the ImageNet crops.

    python src/test_corruption.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from bench.config import Run
from bench.tasks import afc_recognition_eval, pam_eval, serial_order_eval
from src import corruption
from src.stimuli import ImageNetConceptsDataset


def sample_images():
    ds = ImageNetConceptsDataset()
    return [ds.get_image(i) for i in (0, len(ds) // 2, len(ds) - 1)]


def diff(a, b):
    return np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean()


def test_every_corruption_keeps_size_and_changes_pixels(images):
    for img in images:
        for name in corruption.NAMES:
            for s in corruption.SEVERITIES:
                out = corruption.corrupt(img, name, s, seed=0)
                assert out.size == img.size and out.mode == "RGB", (name, s, out.size, out.mode)
                assert diff(out, img) > 0, (name, s)


def test_severity_is_monotone(images):
    # Distance from the original grows with severity for every ImageNet-C type. patch_shuffle moves content
    # rather than degrading it, and noise_image shares nothing with the original, so neither is ordered by
    # severity. Phase scrambling saturates: past a mixing weight of ~0.6 the image is unrelated to the original
    # and the distance depends on the draw, so it is checked up to there and for staying scrambled after.
    for img in images:
        for name in corruption.NAMES:
            if name in ("patch_shuffle", "noise_image"):
                continue
            d = [diff(corruption.corrupt(img, name, s, seed=0), img) for s in corruption.SEVERITIES]
            if name == "phase_scramble":
                assert d[0] < d[1] < d[2] and min(d[3:]) > d[1], (name, d)
            else:
                assert all(x < y for x, y in zip(d, d[1:])), (name, d)


def test_deterministic_per_image(images):
    img = images[0]
    for name in corruption.NAMES:
        a = corruption.corrupt(img, name, 3, seed=0)
        b = corruption.corrupt(img.copy(), name, 3, seed=0)
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
    # The random types draw fresh noise for another seed or severity, and for another image.
    for name in ("gaussian_noise", "shot_noise", "impulse_noise", "patch_shuffle", "phase_scramble", "noise_image"):
        a = corruption.corrupt(img, name, 3, seed=0)
        assert not np.array_equal(np.asarray(a), np.asarray(corruption.corrupt(img, name, 3, seed=1))), name
        assert not np.array_equal(np.asarray(a), np.asarray(corruption.corrupt(images[1], name, 3, seed=0))), name


def test_patch_shuffle_moves_content(images):
    img = images[0]
    for s in corruption.SEVERITIES:
        out = corruption.corrupt(img, "patch_shuffle", s, seed=0)
        # Same pixels, other places.
        assert np.array_equal(np.sort(np.asarray(img).ravel()), np.sort(np.asarray(out).ravel())), s
        assert not np.array_equal(np.asarray(img), np.asarray(out)), s


def test_phase_scramble_keeps_amplitude_spectrum(images):
    img = images[0]
    out = corruption.corrupt(img, "phase_scramble", 5, seed=0)
    a = np.abs(np.fft.fft2(np.asarray(img, dtype=np.float32)[..., 1]))
    b = np.abs(np.fft.fft2(np.asarray(out, dtype=np.float32)[..., 1]))
    # Equal up to the 8-bit rounding and clipping of the output.
    assert np.corrcoef(np.log1p(a).ravel(), np.log1p(b).ravel())[0, 1] > 0.95


def test_noise_image_is_block_noise(images):
    img = images[0]
    for s, block in zip(corruption.SEVERITIES, corruption.PARAMS["noise_image"]):
        x = np.asarray(corruption.corrupt(img, "noise_image", s, seed=0))
        assert x.shape == (img.size[1], img.size[0], 3)
        # Every block is flat and the block grid is not (the image is not one colour).
        assert np.all(x[:block, :block] == x[0, 0]), s
        assert len(np.unique(x.reshape(-1, 3), axis=0)) > 10, s


def test_bad_arguments(images):
    for args in [("blur", 1), ("gaussian_noise", 0), ("gaussian_noise", 6)]:
        try:
            corruption.corrupt(images[0], *args, seed=0)
        except ValueError:
            continue
        raise AssertionError(args)


def test_variants():
    assert corruption.variants({}, "imagenet") == [{}]
    assert corruption.variants({}, "tokens") == [{}]
    assert corruption.variants({"corruptions": ["gaussian_noise"]}, "imagenet") == [
        {"corruption": "gaussian_noise", "severity": s} for s in (1, 2, 3, 4, 5)]
    assert corruption.variants({"corruptions": ["pixelate", "contrast"], "severities": [1, 5]}, "things") == [
        {"corruption": "pixelate", "severity": 1}, {"corruption": "pixelate", "severity": 5},
        {"corruption": "contrast", "severity": 1}, {"corruption": "contrast", "severity": 5}]
    for cfg, dataset in [({"severities": [1]}, "imagenet"),
                         ({"corruptions": ["gaussian_noise"]}, "tokens"),
                         ({"corruptions": ["motion_blur"]}, "imagenet"),
                         ({"corruptions": ["gaussian_noise"], "severities": [0, 1]}, "imagenet")]:
        try:
            corruption.variants(cfg, dataset)
        except ValueError:
            continue
        raise AssertionError((cfg, dataset))


def test_apply_corrupts_every_image_once(images):
    a, b, c = images
    condition = {"corruption": "gaussian_noise", "severity": 2}
    # Recognition/serial order shape: the tested study image reappears at test as a fresh copy.
    trial = {"study_sequence": [a, b], "test_phase": [{"images": [b.copy(), c], "target": 1}]}
    out = corruption.apply(trial, condition, seed=0)
    study, test = out["study_sequence"], out["test_phase"][0]["images"]
    assert np.array_equal(np.asarray(study[1]), np.asarray(test[0]))
    assert np.array_equal(np.asarray(study[0]), np.asarray(corruption.corrupt(a, "gaussian_noise", 2, 0)))
    assert diff(test[1], c) > 0
    # PAM shape: (image, word) pairs and a single cue image.
    trial = {"study_sequence": [(a, "apple"), (b, "bread")], "test_phase": [{"image": a.copy(), "target": "apple"}]}
    out = corruption.apply(trial, condition, seed=0)
    assert out["study_sequence"][0][1] == "apple" and out["study_sequence"][1][1] == "bread"
    assert np.array_equal(np.asarray(out["study_sequence"][0][0]), np.asarray(out["test_phase"][0]["image"]))
    # No corruption in the condition: nothing is touched.
    trial = {"study_sequence": [a], "test_phase": [{"images": [a, b]}]}
    assert corruption.apply(trial, {"dataset": "imagenet"}, seed=0)["study_sequence"][0] is a


def test_expand_conditions_and_filenames():
    base = {"task": "afc_recognition", "prompts": ["original"], "datasets": ["imagenet"], "foil_types": ["novel"],
            "single_foil": True, "n_images": [5], "n_trials": 2}
    assert "corruption" not in afc_recognition_eval.expand_conditions(base)[0]
    conds = afc_recognition_eval.expand_conditions({**base, "corruptions": ["jpeg_compression"], "severities": [3]})
    assert len(conds) == 1 and conds[0]["corruption"] == "jpeg_compression" and conds[0]["severity"] == 3
    name = Run(model={"name": "m"}, task="afc_recognition", condition=conds[0], seed=0).result_name
    assert "corruption-jpeg_compression" in name and "severity-3" in name, name

    conds = serial_order_eval.expand_conditions({"task": "serial_order", "prompts": ["original"], "datasets": ["imagenet"],
                                                 "n_images": [5], "corruptions": ["contrast"], "severities": [1, 2]})
    assert [c["severity"] for c in conds] == [1, 2]
    conds = pam_eval.expand_conditions({"task": "paired_associates", "prompts": ["sandwich"], "datasets": ["imagenet"],
                                        "wordpool": "extended", "n_images": [5], "corruptions": ["pixelate"]})
    assert len(conds) == 5 and all(c["corruption"] == "pixelate" for c in conds)

    # Token items refuse the axis in every task that takes them.
    token = {"prompts": ["original"], "datasets": ["tokens"], "foil_types": ["novel"], "single_foil": True,
             "n_images": [5], "item_lengths": [8], "token_pools": ["clean"], "corruptions": ["gaussian_noise"]}
    for expand, cfg in [(afc_recognition_eval.expand_conditions, {**token, "task": "afc_recognition"}),
                        (serial_order_eval.expand_conditions, {**token, "task": "serial_order"}),
                        (pam_eval.expand_conditions, {**token, "task": "paired_associates", "prompts": ["sandwich"]})]:
        try:
            expand(cfg)
        except ValueError as e:
            assert "corruptions" in str(e), e
            continue
        raise AssertionError(cfg["task"])


if __name__ == "__main__":
    images = sample_images()
    test_every_corruption_keeps_size_and_changes_pixels(images)
    test_severity_is_monotone(images)
    test_deterministic_per_image(images)
    test_patch_shuffle_moves_content(images)
    test_phase_scramble_keeps_amplitude_spectrum(images)
    test_noise_image_is_block_noise(images)
    test_bad_arguments(images)
    test_variants()
    test_apply_corrupts_every_image_once(images)
    test_expand_conditions_and_filenames()
    print("all corruption checks passed")
