"""Build the ImageNet stimulus set: one 256x256 center crop per ImageNet-21k (winter21) class.

Streams shards of timm/imagenet-w21-wds in order (the shards are pre-shuffled, so each holds a few images
of most classes) without storing them, and keeps the first image per class whose short side is >= 256 px
(smaller ones would need upscaling). The short side is resized to 256 and the center cropped, so every
stimulus is 66 tokens. All classes are kept, not only the selected concepts, so the concept selection
(scripts/select_imagenet_concepts.py) can be rerun on whichever classes have an image.

Blank images (Flickr's "photo no longer available" placeholder, all-black frames) are skipped too.
--redo-blank applies that check to crops saved before it existed: it deletes the blank ones and streams
shards 0..N-1 again for those classes only, which gives each class the image a build from scratch would.

Resumable: shards already listed in manifest.jsonl are skipped, by --redo-blank too. Which
image a class gets depends only on shard order, so a rerun from scratch reproduces the set.
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import urllib3
from huggingface_hub import get_token, hf_hub_url
from PIL import Image

REPO_ID = "timm/imagenet-w21-wds"
SIZE = 256
RETRIES = 5


def center_crop(image, size=SIZE):
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.LANCZOS)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def is_blank(image):
    # The placeholder is 99% white (std 9.7 on the crop); real photos, including product shots on white,
    # are std >= 17, and only 0.1% of all crops fall below 9.7.
    a = np.asarray(image.convert("L"), dtype=np.float32)
    return a.std() < 10 and (a.mean() > 240 or a.mean() < 15)


def crop_sample(meta, jpg):
    """The stimulus for one sample, or the reason it is skipped."""
    if min(meta["width"], meta["height"]) < SIZE:
        return None, "small"
    crop = center_crop(Image.open(io.BytesIO(jpg)).convert("RGB"))
    return (None, "blank") if is_blank(crop) else (crop, None)


def stream_samples(shard, max_samples=None):
    """Yield (key, meta, jpg_bytes) from one shard; webdataset members of a sample are adjacent."""
    url = hf_hub_url(REPO_ID, f"imagenet_w21-train-{shard:04d}.tar", repo_type="dataset")
    with requests.get(url, headers={"Authorization": f"Bearer {get_token()}"}, stream=True, timeout=120) as r:
        r.raise_for_status()
        sample, n = {}, 0
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for member in tf:
                key, ext = member.name.rsplit(".", 1)
                if sample and sample["key"] != key:
                    yield sample["key"], sample.get("json"), sample.get("jpg")
                    n += 1
                    if max_samples and n >= max_samples:
                        return
                    sample = {}
                sample["key"] = key
                if ext in ("json", "jpg"):
                    data = tf.extractfile(member).read()
                    sample[ext] = json.loads(data) if ext == "json" else data
            if sample:
                yield sample["key"], sample.get("json"), sample.get("jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory: <wnid>.jpg + manifest.jsonl")
    ap.add_argument("--shards", type=int, default=128, help="stream shards 0..N-1")
    ap.add_argument("--targets", help="concepts JSON from select_imagenet_concepts.py, for coverage logging")
    ap.add_argument("--max-samples", type=int, help="per shard; for a quick test only")
    ap.add_argument("--redo-blank", action="store_true", help="replace saved crops that are blank")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done_shards, have, redo, redo_done = set(), set(), set(), set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            rec = json.loads(line)
            if rec["type"] == "shard_done":
                done_shards.add(rec["shard"])
            elif rec["type"] == "image":
                have.add(rec["wnid"])
                redo.discard(rec["wnid"])
            elif rec["type"] == "redo":
                have.discard(rec["wnid"])
                redo.add(rec["wnid"])
                redo_done.clear()  # a newly added class has not been looked for in any shard yet
            elif rec["type"] == "redo_shard_done":
                redo_done.add(rec["shard"])
    targets = set()
    if args.targets:
        targets = {c["wnid"] for c in json.loads(Path(args.targets).read_text())["concepts"]}

    with manifest.open("a") as log:
        if not manifest.stat().st_size:
            log.write(json.dumps({"type": "config", "repo": REPO_ID, "size": SIZE, **vars(args)}) + "\n")
        if args.redo_blank:
            redo_blank(out, log, have, redo, redo_done, args)
            return
        for shard in range(args.shards):
            if shard in done_shards:
                continue
            t0 = time.time()
            seen, skipped = scan_shard(shard, out, log, have, lambda wnid: wnid not in have, args.max_samples)
            log.write(json.dumps({"type": "shard_done", "shard": shard, "samples": seen}) + "\n")
            log.flush()
            covered = f", targets {len(have & targets)}/{len(targets)}" if targets else ""
            print(f"shard {shard}: {seen} samples, {skipped['small']} too small, {skipped['blank']} blank, "
                  f"{time.time() - t0:.0f}s; classes with an image {len(have)}{covered}", flush=True)


def scan_shard(shard, out, log, have, wanted, max_samples=None):
    """scan_shard_once, retried: the Hub connection drops mid-shard about once an hour. A retry rescans the
    shard from the start; classes saved before the drop are in `have` by then, so the result is the same
    as an uninterrupted scan (only the skip counts include the rescanned part twice)."""
    for attempt in range(1, RETRIES + 1):
        try:
            return scan_shard_once(shard, out, log, have, wanted, max_samples)
        except (requests.RequestException, urllib3.exceptions.HTTPError, tarfile.TarError) as e:
            if attempt == RETRIES:
                raise
            log.flush()
            print(f"shard {shard}: attempt {attempt} failed ({type(e).__name__}), retrying", flush=True)
            time.sleep(30 * attempt)


def scan_shard_once(shard, out, log, have, wanted, max_samples=None):
    """Save the first usable image of each wanted class in this shard; returns (samples, skip counts)."""
    seen, skipped = 0, {"small": 0, "blank": 0}
    for key, meta, jpg in stream_samples(shard, max_samples):
        seen += 1
        if meta is None or jpg is None or not wanted(meta["class_name"]):
            continue
        crop, reason = crop_sample(meta, jpg)
        if crop is None:
            skipped[reason] += 1
            continue
        wnid = meta["class_name"]
        crop.save(out / f"{wnid}.jpg", quality=95)
        have.add(wnid)
        log.write(json.dumps({"type": "image", "wnid": wnid, "key": key, "shard": shard,
                              "width": meta["width"], "height": meta["height"]}) + "\n")
    return seen, skipped


def redo_blank(out, log, have, redo, redo_done, args):
    for wnid in sorted(have):
        path = out / f"{wnid}.jpg"
        if is_blank(Image.open(path)):
            path.unlink()
            have.discard(wnid)
            redo.add(wnid)
            log.write(json.dumps({"type": "redo", "wnid": wnid, "reason": "blank"}) + "\n")
            redo_done.clear()
    log.flush()
    print(f"replacing {len(redo)} blank crops: {' '.join(sorted(redo))}", flush=True)
    for shard in range(args.shards):
        if not redo:
            break
        if shard in redo_done:
            continue
        t0 = time.time()
        found = set(redo)
        scan_shard(shard, out, log, have, lambda wnid: wnid in redo and wnid not in have, args.max_samples)
        redo -= have
        log.write(json.dumps({"type": "redo_shard_done", "shard": shard}) + "\n")
        log.flush()
        print(f"redo shard {shard}: replaced {len(found - redo)}, {len(redo)} left, {time.time() - t0:.0f}s",
              flush=True)
    print(f"no usable image in shards 0..{args.shards - 1}: {' '.join(sorted(redo)) or 'none'}", flush=True)


if __name__ == "__main__":
    main()
