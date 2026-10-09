"""What a trial stores about its thinking trace, and the per-cell summary of those traces."""


def thinking_record(thought):
    """The trace without its `prefill`, which only repeats it with the closing tag."""
    return {k: thought[k] for k in ("trace", "n_tokens", "stop", "seed")}


def thinking_metrics(trials):
    """Trace lengths and how traces ended; empty with thinking off.

    `thinking_frac_budget` matters most: those trials were answered after a force-closed trace,
    so a high value means the cell measures the budget as much as the model.
    """
    records = [t[k] for t in trials for k in ("thinking", "thinking_swapped") if k in t]
    if not records:
        return {}
    lengths = sorted(r["n_tokens"] for r in records)
    return {
        "thinking_n_traces": len(records),
        "thinking_mean_tokens": sum(lengths) / len(lengths),
        "thinking_median_tokens": lengths[len(lengths) // 2],
        "thinking_max_tokens": lengths[-1],
        "thinking_frac_budget": sum(r["stop"] == "budget" for r in records) / len(records),
        "thinking_frac_eos": sum(r["stop"] == "eos" for r in records) / len(records),
    }
