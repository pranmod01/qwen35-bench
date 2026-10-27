"""Download WikiText-103 (raw) and write it as plain text for tokens/corpus.py.

WikiText is Moses-tokenised ("the vote , until" and " @-@ " for hyphens), which is out of distribution for a model
trained on raw text, so paragraphs are detokenised with the small rule set below. Headings are kept verbatim as
document boundaries ("= Title =" opens a document). Writes tokens/corpus.CORPUS (~540 MB) under ZROOT, off the home
quota. Run on the login node, where the internet is:

    source $ZROOT/envs/qwen35/bin/activate && HF_HOME=$ZROOT/hf_cache python scripts/fetch_wikitext.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokens.corpus import CORPUS

RULES = [
    (re.compile(r" @(.)@ "), r"\1"),                       # " @-@ ", " @,@ ", " @.@ " joined numbers and hyphens
    (re.compile(r" ([,.;:!?%)\]}])"), r"\1"),             # no space before closing punctuation
    (re.compile(r"([(\[{$£€]) "), r"\1"),                 # none after opening punctuation
    (re.compile(r" (n't|'t|'s|'m|'re|'ve|'ll|'d)\b"), r"\1"),  # clitics ("didn 't", "it 's")
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),                # quoted phrases: " x " -> "x"
    (re.compile(r" ' ([^']*?) ' "), r" '\1' "),
]


def detokenize(line):
    for pattern, repl in RULES:
        line = pattern.sub(repl, line)
    return line


def main():
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_paras = 0
    with open(CORPUS, "w", encoding="utf-8") as out:
        for row in rows:
            line = row["text"].strip()
            if not line:
                continue
            if line.startswith("= ") and line.endswith(" ="):
                n_docs += not line.startswith("= =")
                out.write(line + "\n\n")
            else:
                n_paras += 1
                out.write(detokenize(line) + "\n\n")
    print(f"wrote {CORPUS}: {n_docs} documents, {n_paras} paragraphs")


if __name__ == "__main__":
    main()
