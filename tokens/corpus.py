"""Natural-text token items: spans of real text, the in-distribution counterpart of the random pools.

A `natural` item is the first L tokens of a sentence from a corpus paragraph, tokenised from the sentence start so
its IDs are exactly what the model would see for that text after a newline (ITEM_SEPARATOR). `scrambled` draws the
same spans and shuffles the tokens within each item, so it has natural's token frequencies with the order, and with it
the meaning, removed. Both are spliced as IDs like the random pools (tokens/stimuli.py).

Old vs new is defined at the span level, not the token level: spans never overlap (at most one per paragraph), but two
spans can share common tokens, so a novel foil differs from the study list in its content words, not in every token.

The corpus is WikiText-103 (scripts/fetch_wikitext.py). Spans are drawn once per item length with a fixed seed into a
cache (CACHE_DIR), which the experiment seeds then sample from:

    HF_HOME=$ZROOT/hf_cache HF_HUB_OFFLINE=1 python -m tokens.corpus --lengths 8 64 256
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np

ZROOT = os.environ.get("ZROOT", "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361")
# The corpus (~540 MB) lives off the home quota; the caches (a few MB per length) live with the other stimuli.
CORPUS = Path(ZROOT) / "datasets/wikitext103/train.txt"
CACHE_DIR = Path(__file__).resolve().parent.parent / "memory_datasets/wikitext103"
TOKENIZER = "Qwen/Qwen3.5-0.8B"   # the tokenizer is shared by all Qwen3.5 sizes
BUILD_SEED = 0
POOLS = ("natural", "scrambled")

# A sentence starts at the paragraph start, or after a sentence end (optionally a closing quote or bracket) and
# whitespace, when the next character is a capital. Abbreviations ("Dr. Smith") give a few false starts; harmless.
SENTENCE_START = re.compile(r"""(?:^|[.!?]["')\]]?\s+)(?=[A-Z])""")


def span_target(item_length):
    """Spans cached per length, at most: 100k for short items, fewer for long ones so no cache exceeds ~32 MB."""
    return min(100_000, 8_000_000 // item_length)


# A pool has to cover the largest cell with room to spare; the L=256 grids stop at 958 items. WikiText-103 has
# only ~2,700 paragraphs with a digit-free run of 256 tokens, so long pools are small and items recur across
# trials, as the image concepts do.
MIN_SPANS = 1_000


def read_paragraphs(path):
    """(document index, paragraph) for every paragraph; headings mark documents and are not paragraphs."""
    doc = -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("= ") and line.endswith(" ="):
                if not line.startswith("= ="):   # a top-level heading opens a document; "= = Section = =" does not
                    doc += 1
                continue
            yield doc, line


def _clean(text):
    # Same rules as stimuli._keep: digits are the answer tokens, a newline is the item boundary.
    return not any(c.isdigit() or c.isspace() and c != " " for c in text) and "�" not in text


def cut_span(tokenizer, paragraph, start, item_length):
    """The first `item_length` tokens of the paragraph from `start`, or None if too short or not clean."""
    limit = 12 * item_length + 64   # characters; well over item_length tokens of English
    tail = paragraph[start:start + limit]
    truncated = len(paragraph) - start > limit
    ids = tokenizer.encode(tail, add_special_tokens=False)
    # A cut mid-word only changes the tokens of that last word, so the span is intact if a token follows it.
    if len(ids) < item_length + (1 if truncated else 0):
        return None
    ids = ids[:item_length]
    return ids if _clean(tokenizer.decode(ids)) else None


def build_cache(tokenizer, item_length, corpus=CORPUS, cache_dir=CACHE_DIR, target=None, seed=BUILD_SEED,
                minimum=MIN_SPANS):
    """Draw spans from random paragraphs (one sentence start each), up to `target`, and write the cache.

    Every paragraph is tried if the corpus has fewer; under `minimum` the pool is refused as too small.
    """
    target = target or span_target(item_length)
    paragraphs = list(read_paragraphs(corpus))
    rng = random.Random(seed)
    order = list(range(len(paragraphs)))
    rng.shuffle(order)
    ids, docs, tried = [], [], 0
    for k in order:
        doc, para = paragraphs[k]
        starts = [m.end() for m in SENTENCE_START.finditer(para)]
        if not starts:
            continue
        tried += 1
        span = cut_span(tokenizer, para, rng.choice(starts), item_length)
        if span is None:
            continue
        ids.append(span)
        docs.append(doc)
        if len(ids) == target:
            break
    if len(ids) < minimum:
        raise ValueError(f"only {len(ids)} clean spans of {item_length} tokens in {corpus}; need at least {minimum}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / f"spans_L{item_length}.npz", ids=np.array(ids, dtype=np.int32), docs=np.array(docs, dtype=np.int32))
    meta = {"corpus": str(corpus), "tokenizer": tokenizer.name_or_path, "vocab_size": len(tokenizer), "seed": seed,
            "item_length": item_length, "n_spans": len(ids), "paragraphs_tried": tried,
            "n_paragraphs": len(paragraphs), "n_documents": max(d for d, _ in paragraphs) + 1}
    (cache_dir / f"spans_L{item_length}.json").write_text(json.dumps(meta, indent=1))
    return meta


def read_meta(item_length, cache_dir=None):
    path = Path(cache_dir or CACHE_DIR) / f"spans_L{item_length}.json"
    if not path.exists():
        raise FileNotFoundError(f"no span cache for item_length {item_length} at {path}; build it with "
                                f"`python -m tokens.corpus --lengths {item_length}` (see tokens/corpus.py)")
    return json.loads(path.read_text())


def n_spans(item_length, cache_dir=None):
    """Pool size for one length, readable before any tokenizer loads (the runner's size check)."""
    return read_meta(item_length, cache_dir)["n_spans"]


class SpanPool:
    """The cached spans for one pool name, loaded per item length on first use."""

    def __init__(self, name, tokenizer, cache_dir=None):
        if name not in POOLS:
            raise ValueError(f"unknown span pool {name!r}; expected one of {POOLS}")
        self.name = name
        self.scrambled = name == "scrambled"
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self._spans = {}

    def spans(self, item_length):
        if item_length not in self._spans:
            meta = read_meta(item_length, self.cache_dir)
            if meta["vocab_size"] != len(self.tokenizer):
                raise ValueError(f"span cache was built for a {meta['vocab_size']}-token vocabulary "
                                 f"({meta['tokenizer']}), this tokenizer has {len(self.tokenizer)}")
            ids = np.load(self.cache_dir / f"spans_L{item_length}.npz")["ids"]
            # A span starts a sentence, so its opening tokens re-encode to themselves; a different tokenizer
            # with the same vocabulary size would not pass.
            first = [int(i) for i in ids[0][:2]]
            if self.tokenizer.encode(self.tokenizer.decode(first), add_special_tokens=False) != first:
                raise ValueError("span cache does not round-trip through this tokenizer")
            self._spans[item_length] = ids
        return self._spans[item_length]

    def sample(self, n_items, item_length):
        """`n_items` distinct spans; scrambled pools shuffle each span's tokens. Uses the global `random`."""
        spans = self.spans(item_length)
        if n_items > len(spans):
            raise ValueError(f"{n_items} items asked of a {self.name} pool with {len(spans)} spans of {item_length}")
        items = []
        for k in random.sample(range(len(spans)), n_items):
            ids = [int(i) for i in spans[k]]
            if self.scrambled:
                random.shuffle(ids)
            items.append(tuple(ids))
        return items


def main():
    parser = argparse.ArgumentParser(description="Build the natural-text span caches.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 64, 256])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for L in args.lengths:
        meta = build_cache(tokenizer, L, corpus=args.corpus, cache_dir=args.cache_dir)
        print(f"L={L}: {meta['n_spans']} spans from {meta['paragraphs_tried']} paragraphs tried "
              f"({meta['n_paragraphs']} paragraphs, {meta['n_documents']} documents)")
        sample = np.load(args.cache_dir / f"spans_L{L}.npz")["ids"][:3]
        for span in sample:
            print("   ", repr(tokenizer.decode([int(i) for i in span])))


if __name__ == "__main__":
    main()
