"""Build the extended PAM word pool for the ImageNet runs (2026-10-16).

The WAS-normed pool (memory_datasets/wasnorm_wordpool.txt, 1,638 words) caps paired associates at 1,638
pairs, which is the Brady/THINGS ceiling anyway but well short of the ~3,500 ImageNet concepts that fit the
context. This pool is the WAS pool plus concrete English nouns matched to it in frequency, until it holds
--size words, so an ImageNet list can draw its words the same way (random.sample from one pool). Brady and
THINGS keep the WAS pool; a pool is chosen per task in the config (`wordpool:`), not here.

Candidates and filters, in order (counts of survivors go in the .json):
  1. WordNet 3.0 noun lemmas that are one lowercase alphabetic word of --min-len to --max-len letters.
     WordNet writes proper nouns capitalised, so this drops them in their proper-noun spelling.
  2. Concrete: the lemma's first noun sense (WordNet orders senses by frequency) is under object.n.01
     (physical objects: organisms, artifacts, places) or matter.n.03 (substances). The WAS words are concrete
     nouns (ACORN, BEAKER, ACID), and recall differs for abstract words. physical_entity.n.01 is too wide:
     it also holds processes ("absorption", "scanning").
  3. Noun in practice, not only in WordNet: more noun synsets than verb, adjective and adverb synsets
     together. Without this the most frequent "concrete nouns" were "have" (a wealthy person), "see" (a
     bishop's seat), "still", "great" and "come".
  4. Not a name: no synset of the word has a capitalised lemma ("john" the toilet also names the apostle;
     "chad", "derby", "titan"), and the word is not in NLTK's first-name list ("mike", "molly").
     Frequency counts are case-blind, so a name's frequency would otherwise carry its common-noun sense in.
  5. Base form: WordNet's morphology finds no other base form for the word, which drops plurals and other
     inflections ("parts", "eggs", "oxen") that would double a singular already in the pool.
  6. Not already in the WAS pool, not naming a pictured concept (every single-word lemma of a selected
     ImageNet concept and the head word of every multi-word lemma), and not in --exclude, a hand-kept list
     (profanity, slurs, drug and crime words, brand names, clippings) kept in the repo.
  7. One word per meaning: a word that shares any noun synset with a WAS word or an earlier-ranked candidate
     is dropped, and so is a British spelling whose American form WordNet also lists ("colour", "theatre",
     "counsellor"). Spelling pairs and close synonyms would otherwise be scored as errors against each other
     in a recall test, and the pool has candidates to spare.
  8. Frequency matched to the WAS pool: wordfreq Zipf frequency between the WAS pool's rarest word and its
     95th percentile, ranked by distance from the WAS median (ties alphabetical) and taken until the pool is
     full, so the rare end is used only as far as needed. Taking the most frequent words instead front-loads
     polysemous words ("state", "end", "side", "area"). The band reaches the WAS minimum because the WAS pool
     and the pictured concepts between them already hold most common concrete nouns: only ~1,600 candidates
     sit above the WAS 5th percentile and ~2,300 above its 1st.
Deterministic: no randomness, so the pool is a function of the inputs and this script.

Run outside the bench env, as scripts/select_imagenet_concepts.py does:
    UV_CACHE_DIR=$ZROOT/uv_cache ~/.local/bin/uv run --no-project --with nltk==3.9.1 --with wordfreq==3.1.1 \
        python scripts/build_pam_wordpool.py
Writes memory_datasets/pam_wordpool_extended.txt (one word per line, upper case, alphabetical, like the WAS
file) and pam_wordpool_extended.json (inputs, filter counts, versions, the added words in rank order).
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import nltk
import wordfreq
from nltk.corpus import wordnet as wn

WAS = Path("memory_datasets/wasnorm_wordpool.txt")
CONCEPTS = Path("memory_datasets/ImageNet21k_w21_256/concepts.json")
OUT = Path("memory_datasets/pam_wordpool_extended")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concept_words(concepts):
    """Words that name a pictured concept: single-word lemmas and the head word of multi-word lemmas."""
    words = set()
    for c in concepts:
        for lemma in c["lemma"].split(","):
            parts = re.split(r"[\s_-]+", lemma.strip().lower())
            if parts and parts[-1]:
                words.add(parts[-1])
    return words


def zipf(word):
    return wordfreq.zipf_frequency(word, "en")


# British -> American spellings, enough for WordNet's variant pairs; a word is a variant if the rewritten form
# is a different WordNet lemma.
SPELLING_RULES = [(r"our$", "or"), (r"our(?=[a-z])", "or"), (r"re$", "er"), (r"ise(d|r|rs)?$", r"ize\1"),
                  (r"yse$", "yze"), (r"ll(er|or|ing|ery)$", r"l\1"), (r"ae", "e"), (r"oe", "e"),
                  (r"^mould", "mold"), (r"ogh", "og"), (r"ph(ate|ide|ur)$", r"f\1"), (r"sulph", "sulf"),
                  (r"gramme$", "gram"), (r"que$", "ck"), (r"xion$", "ction")]


def american(word, lemmas):
    for pattern, repl in SPELLING_RULES:
        other = re.sub(pattern, repl, word)
        if other != word and other in lemmas:
            return other
    return word


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=4000, help="total words in the pool, WAS included")
    parser.add_argument("--min-len", type=int, default=4,
                        help="shortest added word; 3-letter candidates were mostly clippings (SUB, LAB, MAG)")
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--exclude", default="memory_datasets/pam_wordpool_exclude.txt",
                        help="hand-kept words to leave out, one per line, # comments (missing file: none)")
    parser.add_argument("--out", default=str(OUT), help="output stem; writes <out>.txt and <out>.json")
    args = parser.parse_args()
    nltk.download("wordnet", quiet=True)
    nltk.download("names", quiet=True)
    from nltk.corpus import names

    was = [w.strip() for w in WAS.read_text().splitlines() if w.strip()]
    was_lower = {w.lower() for w in was}
    concepts = json.loads(CONCEPTS.read_text())["concepts"]
    pictured = concept_words(concepts)
    excluded = set()
    if Path(args.exclude).exists():
        excluded = {w.strip().lower() for w in Path(args.exclude).read_text().splitlines()
                    if w.strip() and not w.startswith("#")}
    first_names = {n.lower() for n in names.words()}
    concrete_roots = {wn.synset("object.n.01"), wn.synset("matter.n.03")}

    def first_sense(w):
        return wn.synsets(w, pos="n")[0]

    def concrete(w):
        return bool(concrete_roots & set(first_sense(w).closure(lambda s: s.hypernyms())))

    def noun_dominant(w):
        n_noun = len(wn.synsets(w, pos="n"))
        return n_noun > len(wn.synsets(w)) - n_noun

    def is_name(w):
        return w in first_names or any(l[0].isupper() for s in wn.synsets(w, pos="n") for l in s.lemma_names())

    counts = {"was": len(was), "pictured_words": len(pictured), "excluded_listed": len(excluded)}
    stages = [
        ("wordnet_noun_lemmas", lambda w: True),
        ("single_lowercase_word", lambda w: w.isalpha() and w.islower() and args.min_len <= len(w) <= args.max_len),
        ("concrete_first_sense", concrete),
        ("noun_dominant", noun_dominant),
        ("not_a_name", lambda w: not is_name(w)),
        ("base_form", lambda w: set(wn._morphy(w, "n")) <= {w}),
        ("not_was_pictured_excluded", lambda w: w not in was_lower and w not in pictured and w not in excluded),
    ]
    words = {l for s in wn.all_synsets("n") for l in s.lemma_names()}
    all_lemmas = set(words)
    stages.append(("not_british_variant", lambda w: american(w, all_lemmas) == w))
    for stage, keep in stages:
        words = {w for w in words if keep(w)}
        counts[stage] = len(words)

    was_z = sorted(zipf(w) for w in was_lower)
    lo, median, hi = (was_z[int(len(was_z) * p)] for p in (0.0, 0.5, 0.95))
    band = sorted((w for w in words if lo <= zipf(w) <= hi), key=lambda w: (abs(zipf(w) - median), w))
    counts["in_was_frequency_band"] = len(band)

    # One word per meaning; WAS words claim their senses first.
    taken = {s.name() for w in was_lower for s in wn.synsets(w, pos="n")}
    need = args.size - len(was)
    if need < 0:
        raise SystemExit(f"--size {args.size} is smaller than the WAS pool ({len(was)})")
    chosen, dropped_synonym = [], 0
    for w in band:
        senses = {s.name() for s in wn.synsets(w, pos="n")}
        if senses & taken:
            dropped_synonym += 1
            continue
        taken |= senses
        chosen.append(w)
        if len(chosen) == need:
            break
    counts["dropped_shared_sense"] = dropped_synonym
    counts["added"] = len(chosen)
    counts["total"] = len(was) + len(chosen)
    if len(chosen) < need:
        raise SystemExit(f"only {len(chosen)} candidates survive; need {need}. Lower --size or loosen a filter.")

    pool = sorted(was + [w.upper() for w in chosen])
    assert len(pool) == len(set(pool)) == args.size
    out = Path(args.out)
    out.with_suffix(".txt").write_text("\n".join(pool) + "\n")
    zs = sorted(zipf(w) for w in chosen)
    out.with_suffix(".json").write_text(json.dumps({
        "config": {**vars(args), "was_sha256": sha256(WAS), "concepts_sha256": sha256(CONCEPTS),
                   "exclude_sha256": sha256(args.exclude) if Path(args.exclude).exists() else None,
                   "nltk_version": nltk.__version__, "wordnet_version": wn.get_version(),
                   "wordfreq_version": importlib.metadata.version("wordfreq")},
        "counts": counts,
        "zipf_band": {"low": lo, "median": median, "high": hi},
        "zipf_added": {"min": zs[0], "median": zs[len(zs) // 2], "max": zs[-1]},
        "zipf_was": {"min": was_z[0], "median": median, "max": was_z[-1]},
        "added_in_rank_order": [w.upper() for w in chosen],
    }, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    print("zipf band %.2f-%.2f around %.2f; added words span %.2f-%.2f" % (lo, hi, median, zs[0], zs[-1]))


if __name__ == "__main__":
    main()
