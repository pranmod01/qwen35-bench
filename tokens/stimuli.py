"""Token-sequence stimuli: the text counterpart of an image, for the `tokens` dataset.

An item is a fixed-length sequence of token IDs drawn at random from a filtered slice of the vocabulary.
Like an image's visual tokens, it has no label and no meaning as a whole, and old vs new is defined at the
token-ID level. Items are spliced into the prompt as IDs (see Qwen35Evaluator._splice_token_items): decoding
16 random tokens to text and re-tokenizing gives back different IDs in about half of cases, because
neighbouring pieces merge, so a text round-trip would break the ID-level definition.

Within an episode no token is used twice, so a novel foil shares no token with the study list, and a
recombined foil is made only of studied tokens.

Trial builders mirror tasks/recognition.py and tasks/serial_order_memory.py: the same dict shape, and the
global `random` that the adapters seed. The key "images" is kept so the adapters need no second code path.
"""
import random
import re
from dataclasses import dataclass
from tasks.serial_order_memory import lag_bin

DATASET = "tokens"

# Precedes every item, as the vision-start token does for an image. Newline is excluded from both pools,
# so an item can never contain its own boundary.
ITEM_SEPARATOR = "\n"

# Sizes for the Qwen3.5 tokenizer (shared by 0.8B-27B), needed before any model loads so the runner can
# refuse sizes the pool cannot supply. build_pool checks them against the real filter.
POOL_SIZES = {"alpha": 104164, "clean": 245450}


@dataclass(frozen=True)
class TokenItem:
    ids: tuple

    def __len__(self):
        return len(self.ids)


def _keep(token_text, pool):
    # Digits are the answer tokens ("1"/"2"); a stimulus containing one could prime the forced choice.
    if "�" in token_text or any(c.isdigit() for c in token_text):
        return False
    if pool == "alpha":
        return re.fullmatch(r" ?[A-Za-z]+", token_text) is not None
    if pool == "clean":
        # Nothing whitespace-only and no whitespace inside, so no token can look like an item boundary.
        body = token_text[1:] if token_text.startswith(" ") else token_text
        return bool(body) and not any(c.isspace() for c in body)
    raise ValueError(f"unknown token pool {pool!r}; expected one of {sorted(POOL_SIZES)}")


_POOLS = {}


def build_pool(tokenizer, pool):
    """Sorted token IDs that decode on their own to a clean piece of text (see _keep)."""
    key = (tokenizer.name_or_path, pool)
    if key not in _POOLS:
        special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
        ids = [i for i in range(len(tokenizer)) if i not in special and _keep(tokenizer.decode([i]), pool)]
        if len(ids) != POOL_SIZES[pool]:
            raise ValueError(f"{pool} pool has {len(ids)} tokens, POOL_SIZES says {POOL_SIZES[pool]}: "
                             "a different tokenizer or filter; update POOL_SIZES")
        _POOLS[key] = ids
    return _POOLS[key]


def max_items(pool, item_length):
    return POOL_SIZES[pool] // item_length


def sample_items(pool_ids, n_items, item_length):
    if n_items * item_length > len(pool_ids):
        raise ValueError(f"{n_items} disjoint items of {item_length} tokens need {n_items * item_length} "
                         f"tokens, the pool has {len(pool_ids)}")
    ids = random.sample(pool_ids, n_items * item_length)
    return [TokenItem(tuple(ids[k * item_length:(k + 1) * item_length])) for k in range(n_items)]


def recognition_trials(pool_ids, n_items, item_length, foil_type):
    """One 2-AFC test (studied item vs foil) on a studied item at a random position.

    novel:      the foil is a fresh item, sharing no token with the study list.
    recombined: the foil is the first half of studied item A joined to the second half of studied item B,
                so every foil token was studied and only the binding of tokens into items rejects it.
                The correct option is a third studied item, so neither option shares a half with the other.
    """
    if foil_type == "novel":
        items = sample_items(pool_ids, n_items + 1, item_length)
        study, foil = items[:n_items], items[n_items]
        tested = random.randrange(n_items)
        sources = {}
    elif foil_type == "recombined":
        if n_items < 3:
            raise ValueError("recombined foils need at least 3 studied items")
        study = sample_items(pool_ids, n_items, item_length)
        tested, a, b = random.sample(range(n_items), 3)
        half = item_length // 2
        foil = TokenItem(study[a].ids[:half] + study[b].ids[half:])
        sources = {"foil_head_position": a + 1, "foil_tail_position": b + 1}
    else:
        raise ValueError(f"token stimuli support foil types novel and recombined, not {foil_type!r}")

    correct = study[tested]
    target = 1 if random.random() < 0.5 else 2
    options = [correct, foil] if target == 1 else [foil, correct]
    return {
        "study_prompt": "Here is a sequence of items to remember.",
        "study_sequence": study,
        "test_phase": [{
            "images": options,
            "target": target,
            "type": foil_type,
            "metadata": {"test_position": tested + 1, **sources, "option_ids": [list(o.ids) for o in options]},
        }],
    }


# Serial lags covered by each of tasks/serial_order_memory.lag_bin's bins; the last is open-ended.
LAG_BINS = [(1, 1), (2, 3), (4, 10), (11, 100), (101, None)]


def sample_pair_by_lag(n):
    """One study pair (i, j), i < j, with the distribution of balanced_sample_by_lag(all pairs, 1).

    That function lists every pair, which at n=27,000 is 364M tuples. With one test it picks a lag bin
    uniformly among the non-empty ones and then a pair uniformly within it; drawing the lag in proportion
    to its n - d pairs and then the start position gives the same distribution without the list.
    """
    bins = [(lo, min(hi or n - 1, n - 1)) for lo, hi in LAG_BINS if lo <= n - 1]
    lo, hi = random.choice(bins)
    lags = range(lo, hi + 1)
    d = random.choices(lags, weights=[n - lag for lag in lags])[0]
    i = random.randrange(n - d)
    return i, i + d


def serial_order_trials(pool_ids, n_items, item_length):
    """One test pair, lag-balanced as in AFCSerialOrderMemoryTask with n_tests=1."""
    study = sample_items(pool_ids, n_items, item_length)
    i, j = sample_pair_by_lag(n_items)
    shown = [i, j]
    random.shuffle(shown)
    target = 1 if shown[0] < shown[1] else 2
    lag = abs(i - j)
    return {
        "study_prompt": f"Here is a sequence of {n_items} items to remember in order.",
        "study_sequence": study,
        "test_phase": [{
            "images": [study[k] for k in shown],
            "target": target,
            "metadata": {
                "first_serial_position": min(i, j) + 1,
                "second_serial_position": max(i, j) + 1,
                "serial_lag": lag,
                "lag_bin": lag_bin(lag),
                "distance": lag - 1,
                "left_image_serial_position": shown[0] + 1,
                "right_image_serial_position": shown[1] + 1,
                "option_ids": [list(study[k].ids) for k in shown],
            },
        }],
    }


def encode(evaluator, item):
    if isinstance(item, TokenItem):
        return [{"type": "text", "text": ITEM_SEPARATOR}, evaluator._encode_tokens(item.ids)]
    return [evaluator._encode_image(item)]


def encode_all(evaluator, items):
    return [entry for item in items for entry in encode(evaluator, item)]


def reword(text, items):
    """The image wording with "image" read as "item", so the two modalities share one prompt."""
    if not isinstance(items[0], TokenItem):
        return text
    return text.replace("images", "items").replace("image", "item").replace("Image", "Item")
