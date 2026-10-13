"""Pick mutually distinct ImageNet-21k concepts for the ImageNet stimulus set (one image per concept).

ImageNet classes are WordNet noun synsets, which nest ("dog" > "retriever" > "Labrador retriever"), so a
random draw would put a concept and its own subtype in one list. Rules, applied in order:
  1. drop excluded subtrees (default person.n.01: occupations and roles are not identifiable from a photo);
  2. keep leaves only: classes with no descendant in the class list, so no pair is ancestor/descendant;
  3. keep physical WordNet domains only (--domains; drops acts, attributes, communication, ...), before
     step 4 so an abstract class cannot take a parent slot that a physical sibling could fill;
  4. with --one-per-parent, at most one leaf per immediate hypernym, so sibling subtypes
     ("Labrador retriever", "golden retriever") cannot both appear. Leaves are visited in order of a
     per-class rank, sha256(seed:wnid), so which sibling survives depends on --seed. A rank per class (not
     a shuffle of the whole list) means removing a class changes only its own sibling group: with a shuffle,
     dropping 10 selected classes changed 1,164 of 3,808.

Needs only the class list, no images. Run without touching the bench env:
    UV_CACHE_DIR=$ZROOT/uv_cache ~/.local/bin/uv run --no-project --with nltk==3.9.1 \
        python scripts/select_imagenet_concepts.py --synsets imagenet22k_synsets.txt \
        --lemmas imagenet_synset_to_lemma.txt --one-per-parent --out concepts.json
(class lists from timm/data/_info in huggingface/pytorch-image-models).
"""
import argparse
import collections
import hashlib
import json
import random

import nltk
from nltk.corpus import wordnet as wn

PHYSICAL_DOMAINS = ["noun.artifact", "noun.animal", "noun.plant", "noun.food", "noun.object", "noun.body",
                    "noun.substance"]


def parents(synset):
    return synset.hypernyms() + synset.instance_hypernyms()


def ancestors(synset):
    return set(synset.closure(parents))


def load_classes(path):
    with open(path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    # ImageNet IDs are WordNet 3.0 noun offsets, which is the WordNet nltk ships.
    return {wnid: wn.synset_from_pos_and_offset("n", int(wnid[1:])) for wnid in wnids}


def select(classes, exclude, domains, one_per_parent, seed):
    counts = {"classes": len(classes)}

    excluded = {wn.synset(name) for name in exclude}
    classes = {w: s for w, s in classes.items() if not ({s} | ancestors(s)) & excluded}
    counts["after_exclude"] = len(classes)

    in_list = set(classes.values())
    has_descendant = set()
    for s in in_list:
        has_descendant |= ancestors(s) & in_list
    leaves = {w: s for w, s in classes.items() if s not in has_descendant}
    counts["leaves"] = len(leaves)

    if domains:
        leaves = {w: s for w, s in leaves.items() if s.lexname() in domains}
        counts["after_domains"] = len(leaves)

    if not one_per_parent:
        return leaves, counts

    order = sorted(leaves, key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).hexdigest())
    used_parents, chosen = set(), {}
    for wnid in order:
        ps = set(parents(leaves[wnid]))
        if ps & used_parents:
            continue
        used_parents |= ps
        chosen[wnid] = leaves[wnid]
    counts["one_per_parent"] = len(chosen)
    return chosen, counts


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synsets", required=True, help="one wnid per line")
    ap.add_argument("--lemmas", required=True, help="wnid<TAB>lemmas, for readable names")
    ap.add_argument("--exclude", nargs="*", default=["person.n.01"])
    ap.add_argument("--domains", nargs="*", default=PHYSICAL_DOMAINS, help="WordNet lexnames to keep; empty keeps all")
    ap.add_argument("--one-per-parent", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=40, help="random sample of names to print")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nltk.download("wordnet", quiet=True)
    with open(args.lemmas) as f:
        lemmas = dict(line.rstrip("\n").split("\t", 1) for line in f if "\t" in line)

    chosen, counts = select(load_classes(args.synsets), args.exclude, set(args.domains), args.one_per_parent, args.seed)
    domains = collections.Counter(s.lexname() for s in chosen.values())

    print(json.dumps(counts))
    print("WordNet domains:", ", ".join(f"{d.removeprefix('noun.')} {n}" for d, n in domains.most_common()))
    sample = random.Random(args.seed).sample(sorted(chosen), min(args.show, len(chosen)))
    print("Sample:", "; ".join(lemmas.get(w, chosen[w].name()) for w in sample))

    out = {
        "config": {**vars(args), "synsets_sha256": sha256(args.synsets),
                   "nltk_version": nltk.__version__, "wordnet_version": wn.get_version()},
        "counts": counts,
        "domains": dict(domains.most_common()),
        "concepts": [{"wnid": w, "synset": s.name(), "lemma": lemmas.get(w, ""), "domain": s.lexname(),
                      "parents": [p.name() for p in parents(s)]} for w, s in sorted(chosen.items())],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
