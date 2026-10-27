"""Image corruptions, ImageNet-C style, as a config axis: the image counterpart of the token pools.

The token runs ask whether memory for text depends on the items being in distribution (`clean` random IDs
vs `natural` spans vs `scrambled`). This asks the same of the images: the same stimuli, corrupted at five
severities, so a drop with severity says how much the image results rest on the images being natural.

A task config lists `corruptions` (names below) and `severities` (1-5, default all), crossed with its other
conditions for image datasets; the condition gains `corruption` and `severity`, absent when unset, so the
clean final runs are the reference and their filenames still match. Token items have no pixels, so the
keys are refused for the `tokens` dataset.

Every image is corrupted as a deterministic function of its pixels, the run seed and the severity, so the
copy shown at test is identical to the copy studied, as a token item's IDs are. Noise redrawn at test
would turn recognition into a generalisation test. Corruption is applied to the image as loaded, before
the evaluator's resize; the ImageNet crops are already 256 px, so for them nothing is resampled after.

Parameters follow ImageNet-C (Hendrycks & Dietterich, 2019; the `imagecorruptions` package) for the types
that need only numpy, PIL and scipy. Left out: glass, motion and zoom blur, snow, frost, fog and elastic
(external assets, ImageMagick or per-pixel loops).

Three types are not ImageNet-C and ask a different question: not whether the images must be in distribution
but whether the model remembers images that are not of anything, as the random-ID token items are not.
`patch_shuffle` permutes square blocks of the image (severity: blocks per side, 2 to 32; at 32 the 8 px
blocks are smaller than the 32 px vision-token cell). `phase_scramble` keeps each channel's Fourier
amplitude spectrum and mixes a random phase into its own (severity: the mixing weight, 0.2 to 1.0), so the
colour and spatial-frequency statistics stay natural while the objects dissolve, the usual "scrambled"
control in vision work and the closest analogue of the `scrambled` token pool. `noise_image` replaces the
image with uniform RGB noise, seeded by the original's pixels so each concept maps to its own noise image
and old vs new stays defined (severity: the grain, 32 px blocks down to single pixels), the analogue of the
`clean` random-ID pool.
"""
import io
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

SEVERITIES = (1, 2, 3, 4, 5)

# Per corruption, the parameter at severities 1-5.
PARAMS = {
    "gaussian_noise": [0.08, 0.12, 0.18, 0.26, 0.38],
    "shot_noise": [60, 25, 12, 5, 3],
    "impulse_noise": [0.03, 0.06, 0.09, 0.17, 0.27],
    "defocus_blur": [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)],
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],
    "pixelate": [0.6, 0.5, 0.4, 0.3, 0.25],
    "jpeg_compression": [25, 18, 15, 10, 7],
    "patch_shuffle": [2, 4, 8, 16, 32],
    "phase_scramble": [0.2, 0.4, 0.6, 0.8, 1.0],
    "noise_image": [32, 16, 8, 4, 1],
}
NAMES = tuple(PARAMS)


def _as_float(image):
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_image(x):
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255 + 0.5))


def gaussian_noise(image, c, rng):
    return _to_image(_as_float(image) + rng.normal(0, c, size=image.size[::-1] + (3,)))


def shot_noise(image, c, rng):
    return _to_image(rng.poisson(_as_float(image) * c) / c)


def impulse_noise(image, c, rng):
    """Salt-and-pepper on a fraction c of the values (per channel, as skimage's `s&p` does), half each."""
    x = _as_float(image)
    hit = rng.random(x.shape) < c
    salt = rng.random(x.shape) < 0.5
    x[hit & salt] = 1.0
    x[hit & ~salt] = 0.0
    return _to_image(x)


def defocus_blur(image, c, rng):
    radius, alias_blur = c
    grid = np.arange(-radius, radius + 1)
    kernel = ((grid[:, None] ** 2 + grid[None, :] ** 2) <= radius ** 2).astype(np.float32)
    kernel = ndimage.gaussian_filter(kernel, alias_blur)
    kernel /= kernel.sum()
    # Mirror padding then an FFT convolution: a direct 21x21 convolution took 70 ms per image, more CPU than
    # the model takes per image at large n.
    x = np.pad(_as_float(image), ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    return _to_image(fftconvolve(x, kernel[..., None], mode="valid"))


def contrast(image, c, rng):
    x = _as_float(image)
    means = x.mean(axis=(0, 1), keepdims=True)
    return _to_image((x - means) * c + means)


def brightness(image, c, rng):
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + c * 255, 0, 255)
    return Image.fromarray(np.uint8(hsv), mode="HSV").convert("RGB")


def pixelate(image, c, rng):
    w, h = image.size
    small = image.resize((max(1, int(w * c)), max(1, int(h * c))), Image.Resampling.BOX)
    return small.resize((w, h), Image.Resampling.BOX)


def jpeg_compression(image, c, rng):
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=c)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def patch_shuffle(image, c, rng):
    """Shuffle the c x c grid of square blocks. Any remainder beyond the grid (none at 256 px) stays put."""
    x = np.asarray(image).copy()
    h, w = x.shape[:2]
    bh, bw = h // c, w // c
    if bh == 0 or bw == 0:
        raise ValueError(f"{c}x{c} blocks need an image of at least {c} px a side, got {w}x{h}")
    blocks = [x[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].copy() for i in range(c) for j in range(c)]
    order = rng.permutation(len(blocks))
    for k, src in enumerate(order):
        i, j = divmod(k, c)
        x[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = blocks[src]
    return Image.fromarray(x)


def phase_scramble(image, c, rng):
    """Add c times a random phase to every channel's Fourier phase, keeping the amplitude spectrum.

    The random phase is that of a real random field, so it has the Hermitian symmetry that keeps the inverse
    transform real; one field for all three channels keeps the colour structure consistent across channels.
    """
    x = _as_float(image)
    h, w = x.shape[:2]
    random_phase = np.angle(np.fft.fft2(rng.standard_normal((h, w))))
    # A real field's DC phase is 0 or pi; pi would flip the image's mean to negative and clip it to black.
    random_phase[0, 0] = 0.0
    out = np.empty_like(x)
    for ch in range(3):
        f = np.fft.fft2(x[..., ch])
        out[..., ch] = np.real(np.fft.ifft2(np.abs(f) * np.exp(1j * (np.angle(f) + c * random_phase))))
    return _to_image(out)


def noise_image(image, c, rng):
    """Uniform RGB noise of the image's size in c x c px blocks; any remainder beyond the grid repeats the edge."""
    w, h = image.size
    grid = rng.integers(0, 256, size=(-(-h // c), -(-w // c), 3), dtype=np.uint8)
    return Image.fromarray(np.repeat(np.repeat(grid, c, axis=0), c, axis=1)[:h, :w])


FUNCTIONS = {name: globals()[name] for name in NAMES}


def image_rng(image, seed, severity):
    """A generator fixed by the pixels, so the same image is corrupted the same way every time it appears."""
    return np.random.default_rng([zlib.crc32(image.tobytes()), seed, severity])


def corrupt(image, name, severity, seed):
    if name not in FUNCTIONS:
        raise ValueError(f"unknown corruption {name!r}; expected one of {NAMES}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    image = image.convert("RGB")
    return FUNCTIONS[name](image, PARAMS[name][severity - 1], image_rng(image, seed, severity))


def variants(task_cfg, dataset):
    """`corruption`/`severity` settings to cross with each condition; a single empty variant when unset."""
    names = task_cfg.get("corruptions")
    severities = task_cfg.get("severities")
    if not names:
        if severities:
            raise ValueError("`severities` needs `corruptions`")
        return [{}]
    if dataset == "tokens":
        raise ValueError("`corruptions` is for image datasets; token items have no pixels")
    unknown = set(names) - set(NAMES)
    if unknown:
        raise ValueError(f"unknown corruption(s) {sorted(unknown)}; expected {list(NAMES)}")
    severities = severities or list(SEVERITIES)
    bad = [s for s in severities if s not in SEVERITIES]
    if bad:
        raise ValueError(f"severities must be from {list(SEVERITIES)}, got {bad}")
    return [{"corruption": name, "severity": s} for name in names for s in severities]


def apply(trial_data, condition, seed):
    """Corrupt every image in a task's trial dict in place, under the condition's corruption, if any.

    Handles the three image trial shapes: `study_sequence` of images or (image, word) pairs, and a test
    with `images` (forced choice) or `image` (cue). Images are keyed by pixels so a studied image and its
    copy at test are corrupted once and come out identical.
    """
    name = condition.get("corruption")
    if not name:
        return trial_data
    severity = condition["severity"]
    done = {}

    def one(image):
        key = zlib.crc32(image.tobytes())
        if key not in done:
            done[key] = corrupt(image, name, severity, seed)
        return done[key]

    def each(item):
        return (one(item[0]), *item[1:]) if isinstance(item, tuple) else one(item)

    trial_data["study_sequence"] = [each(item) for item in trial_data["study_sequence"]]
    for test in trial_data.get("test_phase", []):
        if "images" in test:
            test["images"] = [one(img) for img in test["images"]]
        if "image" in test:
            test["image"] = one(test["image"])
    return trial_data
