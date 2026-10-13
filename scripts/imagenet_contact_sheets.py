"""Contact sheets for reviewing the ImageNet stimuli: numbered 160 px thumbnails of every selected concept,
ordered by WordNet domain then name, 192 per sheet. index.tsv maps each number to its wnid, so a reviewer
can name the numbers to exclude. Numbers change whenever concepts.json changes; redraw after every change.

    python scripts/imagenet_contact_sheets.py memory_datasets/ImageNet21k_w21_256
"""
import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from build_imagenet_stimuli import is_blank  # noqa: E402

TILE, LABEL, COLS, ROWS = 160, 34, 12, 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stimuli", help="directory with <wnid>.jpg and concepts.json")
    args = ap.parse_args()

    root = Path(args.stimuli)
    concepts = json.loads((root / "concepts.json").read_text())["concepts"]
    concepts.sort(key=lambda c: (c["domain"], c["lemma"].lower()))
    blank = [c["wnid"] for c in concepts if is_blank(Image.open(root / f"{c['wnid']}.jpg"))]
    if blank:
        raise SystemExit(f"blank crops among the selected concepts: {blank}")

    out = root / "contact_sheets"
    shutil.rmtree(out, ignore_errors=True)  # stale sheets would carry old numbers
    out.mkdir()
    with open(out / "index.tsv", "w") as f:
        f.write("index\twnid\tdomain\tlemma\n")
        for i, c in enumerate(concepts):
            f.write(f"{i}\t{c['wnid']}\t{c['domain'].removeprefix('noun.')}\t{c['lemma']}\n")

    font = ImageFont.load_default(size=12)
    per_sheet = COLS * ROWS
    for start in range(0, len(concepts), per_sheet):
        chunk = concepts[start:start + per_sheet]
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + LABEL)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, c in enumerate(chunk):
            x, y = (k % COLS) * TILE, (k // COLS) * (TILE + LABEL)
            sheet.paste(Image.open(root / f"{c['wnid']}.jpg").resize((TILE - 4, TILE - 4)), (x + 2, y + 2))
            label = textwrap.wrap(f"{start + k} {c['lemma'].split(',')[0]}", 24)[:2]
            draw.multiline_text((x + 3, y + TILE), "\n".join(label), fill="black", font=font, spacing=1)
        name = f"sheet_{start // per_sheet:02d}_{start:04d}-{start + len(chunk) - 1:04d}.jpg"
        sheet.save(out / name, quality=85)
        domains = sorted({c["domain"].removeprefix("noun.") for c in chunk})
        print(name, "/".join(domains))


if __name__ == "__main__":
    main()
