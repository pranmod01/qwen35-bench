#!/usr/bin/env python
"""Contact sheet of every corruption at every severity, on a few ImageNet crops, to eyeball before a run.

One block per image: rows are the corruptions in src/corruption.py, columns the clean crop then severities
1-5. The noise is drawn exactly as a run draws it (per image, seed 0), so the sheet shows the stimuli
a model would see.

    python scripts/corruption_contact_sheet.py [--out plots/corruption/corruption_contact_sheet.png] [--n 3] [--cell 128]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from src import corruption
from src.stimuli import ImageNetConceptsDataset

LABEL_W = 130
HEADER_H = 18


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="plots/corruption/corruption_contact_sheet.png")
    parser.add_argument("--n", type=int, default=3, help="images, spread evenly through concepts.json")
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = ImageNetConceptsDataset()
    picks = [round(k * (len(ds) - 1) / max(1, args.n - 1)) for k in range(args.n)]
    cols = 1 + len(corruption.SEVERITIES)
    cell, gap = args.cell, 4
    block_w = cols * (cell + gap)
    sheet = Image.new("RGB", (LABEL_W + args.n * block_w, HEADER_H + len(corruption.NAMES) * (cell + gap)), "white")
    draw = ImageDraw.Draw(sheet)

    for b, index in enumerate(picks):
        img = ds.get_image(index)
        meta = ds.get_metadata(index)
        x0 = LABEL_W + b * block_w
        draw.text((x0, 2), f"{meta['category']} ({meta['wnid']}): clean, then severity 1-5", fill="black")
        for r, name in enumerate(corruption.NAMES):
            y = HEADER_H + r * (cell + gap)
            if b == 0:
                draw.text((4, y + cell // 2 - 6), name, fill="black")
            for c in range(cols):
                out = img if c == 0 else corruption.corrupt(img, name, c, args.seed)
                sheet.paste(out.resize((cell, cell), Image.Resampling.LANCZOS), (x0 + c * (cell + gap), y))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
