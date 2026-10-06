"""
Visual Haystacks Task

A vision-centric Needle-In-A-Haystack benchmark for evaluating VLM long-context
visual retrieval and reasoning capabilities.

Based on: https://visual-haystacks.github.io/
Dataset: https://huggingface.co/datasets/tsunghanwu/visual_haystacks

Task types:
- Single-needle: "For the image with [anchor], is there [target]?"
- Multi-needle (all): "For all images with [anchor], do all of them contain [target]?"
- Multi-needle (any): "For all images with [anchor], do any of them contain [target]?"
"""
import json
import random
import ssl
import urllib.request
from pathlib import Path
from PIL import Image

# The COCO image CDN has a certificate hostname mismatch on some cluster nodes.
# We use an unverified context so fetches still work; the controlled URL is safe.
_SSL_UNVERIFIED_CTX = ssl.create_default_context()
_SSL_UNVERIFIED_CTX.check_hostname = False
_SSL_UNVERIFIED_CTX.verify_mode = ssl.CERT_NONE

COCO_IMAGE_BASE_URL = "https://images.cocodataset.org"

# Map split name -> COCO subdirectory
_SPLIT_TO_SUBDIR = {
    "train2017": "train2017",
    "val2017": "val2017",
    "test2017": "test2017",
}


def _resolve_image_path(image_root: Path, image_path: str) -> Path:
    """Resolve a VHS image path (e.g. 'val2017/000000123.jpg') under image_root."""
    return image_root / image_path


def _fetch_image(
    image_root: Path,
    image_path: str,
    coco_base_url: str = COCO_IMAGE_BASE_URL,
    fetch_timeout_s: int = 120,
) -> None:
    """Download a single COCO image into image_root if it doesn't exist."""
    dest = _resolve_image_path(image_root, image_path)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{coco_base_url.rstrip('/')}/{image_path}"
    with urllib.request.urlopen(url, context=_SSL_UNVERIFIED_CTX, timeout=fetch_timeout_s) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def iter_unique_image_paths_in_vhs_qa(qa_root) -> list:
    """Return sorted list of unique image paths referenced across all VHS QA JSONs."""
    qa_root = Path(qa_root)
    paths = set()
    for json_file in qa_root.glob("visual_haystack_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        for item in data:
            for p in item.get("image", []):
                paths.add(p)
    return sorted(paths)


def prefetch_vhs_coco_images(
    qa_root,
    image_root,
    coco_base_url: str = COCO_IMAGE_BASE_URL,
    fetch_timeout_s: int = 120,
    skip_existing: bool = True,
) -> dict:
    """Download every COCO image referenced in VHS QA JSONs.

    Returns a stats dict with counts of downloaded, skipped, and failed images.
    """
    image_root = Path(image_root)
    unique_paths = iter_unique_image_paths_in_vhs_qa(qa_root)
    stats = {"total": len(unique_paths), "downloaded": 0, "skipped": 0, "failed": 0}
    for image_path in unique_paths:
        dest = _resolve_image_path(image_root, image_path)
        if skip_existing and dest.exists():
            stats["skipped"] += 1
            continue
        try:
            _fetch_image(image_root, image_path, coco_base_url, fetch_timeout_s)
            stats["downloaded"] += 1
        except Exception as e:
            print(f"Failed to fetch {image_path}: {e}")
            stats["failed"] += 1
    return stats


def _build_qa_filename(mode: str, split: str, image_count: str) -> str:
    """Build the expected VHS QA JSON path (relative to qa_root) from eval args.

    Actual layout on disk:
      multi_needle/visual_haystack_{image_count}.json
      single_needle/{split}/visual_haystack_{image_count}.json
    """
    if mode == "single_needle":
        return f"single_needle/{split}/visual_haystack_{image_count}.json"
    elif mode == "multi_needle":
        return f"multi_needle/visual_haystack_{image_count}.json"
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'single_needle' or 'multi_needle'.")


class VisualHaystacksTask:
    """Visual Haystacks task loaded from local QA JSON files + COCO images."""

    def __init__(
        self,
        qa_root: str = "datasets/VHs_qa",
        image_root: str = "datasets/coco",
        mode: str = "single_needle",
        split: str = "VHs_large",
        image_count: str = "10",
        max_samples: int = None,
        shuffle_images: bool = True,
        seed: int = 0,
        fetch_missing_coco: bool = False,
        coco_base_url: str = COCO_IMAGE_BASE_URL,
        fetch_timeout_s: int = 120,
    ):
        self.qa_root = Path(qa_root)
        self.image_root = Path(image_root)
        self.mode = mode
        self.split = split
        self.image_count = image_count
        self.max_samples = max_samples
        self.shuffle_images = shuffle_images
        self.seed = seed
        self.fetch_missing_coco = fetch_missing_coco
        self.coco_base_url = coco_base_url
        self.fetch_timeout_s = fetch_timeout_s

        self._rng = random.Random(seed)
        self._data = []
        self._load_data()

    def _load_data(self):
        filename = _build_qa_filename(self.mode, self.split, self.image_count)
        qa_file = self.qa_root / filename
        if not qa_file.exists():
            # Collect valid sizes from files actually present for this mode/split
            if self.mode == "multi_needle":
                search_dir = self.qa_root / "multi_needle"
            else:
                search_dir = self.qa_root / "single_needle" / self.split
            valid_sizes = sorted(
                int(p.stem.split("_")[-1])
                for p in search_dir.glob("visual_haystack_*.json")
                if p.stem.split("_")[-1].isdigit()
            ) if search_dir.exists() else []
            raise FileNotFoundError(
                f"VHS QA file not found: {qa_file}\n"
                f"Valid image counts for mode={self.mode!r}"
                + (f", split={self.split!r}" if self.mode == "single_needle" else "")
                + f": {valid_sizes}\n"
                f"Download the Visual Haystacks dataset and place QA JSONs under {self.qa_root}."
            )
        with open(qa_file) as f:
            data = json.load(f)
        if self.max_samples is not None:
            data = data[: self.max_samples]
        self._data = data
        print(f"Loaded {len(self._data)} trials from {qa_file.name}")

    def _load_image(self, image_path: str) -> Image.Image:
        dest = _resolve_image_path(self.image_root, image_path)
        if not dest.exists():
            if self.fetch_missing_coco:
                _fetch_image(self.image_root, image_path, self.coco_base_url, self.fetch_timeout_s)
            else:
                raise FileNotFoundError(
                    f"Image not found: {dest}. "
                    f"Run prefetch_vhs_coco or use --fetch-missing-coco."
                )
        return Image.open(dest).convert("RGB")

    def get_trials(self):
        """Return list of trial dicts with keys: images, prompt, target, metadata."""
        trials = []
        for item in self._data:
            # Oracle format uses pos_image/neg_image; other sizes use image
            if "image" in item:
                image_paths = item["image"]
            else:
                image_paths = item.get("pos_image", []) + item.get("neg_image", [])

            if self.shuffle_images:
                image_paths = list(image_paths)
                self._rng.shuffle(image_paths)

            images = []
            missing = []
            for p in image_paths:
                try:
                    images.append(self._load_image(p))
                except Exception as e:
                    missing.append(p)
                    print(f"Warning: could not load {p}: {e}")

            if missing:
                continue  # skip trials with missing images

            conversations = item.get("conversations", [])
            if len(conversations) < 2:
                continue
            prompt = conversations[0]["value"]
            target = conversations[1]["value"].lower().strip()

            trials.append(
                {
                    "images": images,
                    "prompt": (
                        prompt
                        + " The answer is 'yes' for approximately half of all questions."
                        + " Answer with only 'yes' or 'no'."
                    ),
                    "target": target,
                    "metadata": {
                        "id": item.get("id"),
                        "mode": self.mode,
                        "split": self.split,
                        "image_count": self.image_count,
                        "n_images": len(images),
                    },
                }
            )
        return trials


class VisualHaystacksTaskSimple:
    """Simplified Visual Haystacks using THINGS dataset (no COCO dependency).

    Creates single-needle style questions:
    "For the image containing [category], is there also [other category]?"

    Since THINGS images are single objects, the answer is always based on
    whether the target category appears in the haystack.
    """

    def __init__(
        self, n_images: int = 10, n_trials: int = 20, n_needles: int = 1
    ):
        """
        Args:
            n_images: Total images in haystack
            n_trials: Number of trials
            n_needles: Number of needle images (default 1)
        """
        if n_images < n_needles + 1:
            raise ValueError(
                f"n_images ({n_images}) must be at least n_needles + 1 ({n_needles + 1}) "
                f"to fit both needle(s) and a potential target image."
            )
        self.n_images = n_images
        self.n_trials = n_trials
        self.n_needles = n_needles

        from src.stimuli import ThingsDataset

        # Need enough categories for haystack + needle + target variations
        n_cats = max(n_images * 2, 100)
        self.dataset = ThingsDataset(n_categories=n_cats, exemplars_per_category=1)

    def get_trials(self):
        """Generate Visual Haystacks trials using THINGS."""
        trials = []
        n_cats = len(self.dataset)

        for _ in range(self.n_trials):
            # Select categories
            indices = list(range(n_cats))
            random.shuffle(indices)

            # Needle category (the "anchor" - what we're looking for)
            needle_idx = indices[0]
            needle_cat = self.dataset.category_names[needle_idx]

            # Target category (what we ask about)
            target_idx = indices[1]
            target_cat = self.dataset.category_names[target_idx]

            # Decide if target should be present (50/50)
            target_present = random.random() < 0.5

            # Build haystack — always exactly n_images images
            images = []

            # Add needle image(s)
            for i in range(self.n_needles):
                images.append(self.dataset.get_image(needle_idx, 0))

            # Add target if present, otherwise add an extra distractor in its slot
            start_idx = 2
            if target_present:
                images.append(self.dataset.get_image(target_idx, 0))
                start_idx = 3

            # Fill remaining slots with distractors (not needle or target)
            n_distractors = self.n_images - len(images)
            distractor_indices = indices[start_idx : start_idx + n_distractors]

            for idx in distractor_indices:
                images.append(self.dataset.get_image(idx, 0))

            assert len(images) == self.n_images, (
                f"Haystack size mismatch: expected {self.n_images}, got {len(images)}"
            )

            random.shuffle(images)

            # Create question
            question = f"In the set of images shown, there is an image of a {needle_cat}. Is there also a separate image of a {target_cat} anywhere in the set?"
            answer = "yes" if target_present else "no"

            trials.append(
                {
                    "images": images,
                    "prompt": question + " Answer with only 'yes' or 'no'.",
                    "target": answer,
                    "metadata": {
                        "needle": needle_cat,
                        "target": target_cat,
                        "target_present": target_present,
                        "n_images": len(images),
                    },
                }
            )

        return trials


if __name__ == "__main__":
    # Test with THINGS-based simple version
    print("Testing VisualHaystacksTaskSimple...")
    task = VisualHaystacksTaskSimple(n_images=5, n_trials=3)
    trials = task.get_trials()

    print(f"Generated {len(trials)} trials")
    for i, t in enumerate(trials):
        print(f"\nTrial {i}:")
        print(f"  Images: {len(t['images'])}")
        print(f"  Question: {t['prompt'][:60]}...")
        print(f"  Answer: {t['target']}")
        print(f"  Metadata: {t['metadata']}")
