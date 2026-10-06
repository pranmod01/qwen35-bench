import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.metrics import f1_score


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion.

    Returns (point, lo, hi). Handles n=0 and edge counts (0/n, n/n) gracefully —
    unlike the bootstrap, which collapses to a point at the boundaries.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(phat), float(max(0.0, center - margin)), float(min(1.0, center + margin))


def bootstrap_ci(data, statistic_fn, n_boot=2000, ci=0.95, seed=0):
    """Percentile bootstrap CI for an arbitrary statistic over a list of items.

    `data` is any indexable collection (list of per-trial dicts, array of floats).
    `statistic_fn` takes a resampled list and returns a scalar.

    Returns (point_estimate, lo, hi).
    """
    if not data:
        return 0.0, 0.0, 0.0
    data = list(data)
    n = len(data)
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(data))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = statistic_fn([data[i] for i in idx])
    # Drop NaN resamples (degenerate cases where the statistic is undefined,
    # e.g. an MST resample with zero lures or zero foils).
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")
    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 - (1 - ci) / 2)
    lo, hi = np.percentile(stats, [lo_pct, hi_pct])
    return point, float(lo), float(hi)

def calculate_metrics(responses, targets):
    """Calculate performance metrics for the continuous recognition task."""
    responses = np.array(responses)
    targets = np.array(targets)

    hits = np.sum((responses == 1) & (targets == 1))
    misses = np.sum((responses == 0) & (targets == 1))
    fas = np.sum((responses == 1) & (targets == 0))
    crs = np.sum((responses == 0) & (targets == 0))

    n_old = hits + misses
    n_new = fas + crs

    accuracy = (hits + crs) / (n_old + n_new) if (n_old + n_new) > 0 else 0
    hit_rate = hits / n_old if n_old > 0 else 0
    fa_rate = fas / n_new if n_new > 0 else 0

    # Standard approximations for d-prime to avoid infinity
    adj_hit_rate = (hits + 0.5) / (n_old + 1) if n_old > 0 else 0.5
    adj_fa_rate = (fas + 0.5) / (n_new + 1) if n_new > 0 else 0.5
    d_prime = norm.ppf(adj_hit_rate) - norm.ppf(adj_fa_rate)

    # Weighted F1
    wf1 = f1_score(targets, responses, average='weighted', zero_division=0)

    return {
        "accuracy": float(accuracy),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(fa_rate),
        "d_prime": float(d_prime),
        "weighted_f1": float(wf1),
        "counts": {"hits": int(hits), "misses": int(misses), "false_alarms": int(fas), "correct_rejections": int(crs)}
    }

def calculate_hit_rate_by_delay(responses, targets, delays):
    """Calculate hit rate for each distinct delay value."""
    responses = np.array(responses)
    targets = np.array(targets)
    delays = np.array(delays)

    old_mask = (targets == 1)
    old_responses = responses[old_mask]
    old_delays = delays[old_mask]

    unique_delays = np.unique([d for d in old_delays if d is not None])
    hr_by_delay = {}

    for d in unique_delays:
        mask = (old_delays == d)
        hr_by_delay[int(d)] = float(np.mean(old_responses[mask]))

    return hr_by_delay

def calculate_serial_order_metrics(reported_positions, actual_positions):
    """Metrics for serial order memory tasks with position reports."""
    reported = np.array(reported_positions)
    actual = np.array(actual_positions)

    errors = np.abs(reported - actual)
    avg_error = np.mean(errors)
    correct = np.sum(errors == 0)

    rho = pval = None
    if len(reported) > 1:
        rho_val, pval_val = spearmanr(reported, actual)
        rho = float(rho_val)
        pval = float(pval_val)

    return {
        "average_error": float(avg_error),
        "n_correct": int(correct),
        "accuracy": float(correct / len(actual)) if len(actual) > 0 else 0,
        "spearman_rho": rho,
        "spearman_pval": pval,
    }

def calculate_afc_serial_order_metrics(results):
    """Metrics for 2-AFC serial order memory using stored probe distances."""
    if not results:
        return {"accuracy": 0, "n_correct": 0, "total": 0, "accuracy_by_distance": {}}

    n_correct = 0
    accuracy_by_distance = {}

    for res in results:
        target = res.get("target")
        reported = res.get("reported")
        metadata = res.get("metadata", {})
        distance = metadata.get("distance")
        correct = int(target == reported)
        n_correct += correct

        if distance is not None:
            accuracy_by_distance.setdefault(int(distance), []).append(correct)

    total = len(results)
    return {
        "accuracy": n_correct / total if total > 0 else 0,
        "n_correct": n_correct,
        "total": total,
        "accuracy_by_distance": {
            distance: float(np.mean(values))
            for distance, values in sorted(accuracy_by_distance.items())
        },
    }

def calculate_color_metrics(reported_colors, actual_colors, n_colors=360):
    """Metrics for continuous color reports on a circular wheel.

    By default, values are interpreted as degrees in [0, 360). Older callers can
    still pass a different n_colors value for discrete indexed reports.
    """
    reported = np.array(reported_colors)
    actual = np.array(actual_colors)

    # Circular error
    diff = (reported - actual + n_colors / 2) % n_colors - n_colors / 2
    abs_error = np.abs(diff)

    # Heuristic guess rate: proportion of errors > threshold
    threshold = n_colors / 4
    guess_rate = np.mean(abs_error > threshold)

    avg_err = float(np.mean(abs_error))
    return {
        "accuracy": max(0.0, 1.0 - avg_err / (n_colors / 2)),
        "average_abs_error": avg_err,
        "guess_rate_heuristic": float(guess_rate),
        "precision_heuristic": float(1.0 / np.std(diff)) if np.std(diff) > 0 else 0
    }

def calculate_named_color_metrics(reported_colors, actual_colors):
    """Metrics for the named color memory task (accuracy, per-color breakdown)."""
    if not reported_colors:
        return {"accuracy": 0, "n_correct": 0, "total": 0, "accuracy_by_color": {}}

    n_correct = 0
    by_color = {}

    for reported, actual in zip(reported_colors, actual_colors):
        correct = int(str(reported).strip().lower() == str(actual).strip().lower())
        n_correct += correct
        by_color.setdefault(str(actual), []).append(correct)

    total = len(reported_colors)
    return {
        "accuracy": n_correct / total if total > 0 else 0,
        "n_correct": n_correct,
        "total": total,
        "accuracy_by_color": {
            color: float(np.mean(vals)) for color, vals in sorted(by_color.items())
        },
    }

def calculate_pam_metrics(results):
    """Calculates accuracy for Paired Associate Memory (word or image-image variants)."""
    if not results:
        return {"accuracy": 0, "n_correct": 0, "total": 0}

    n_correct = 0
    total = len(results)

    for res in results:
        target = str(res.get('target', '')).strip().lower()
        reported = str(res.get('reported', '')).strip().lower()
        if target == reported:
            n_correct += 1

    accuracy = n_correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "total": total
    }


def calculate_2afc_metrics(trials):
    """
    Calculate memory score metrics for 2-AFC recognition task.

    For 2-AFC, d' = sqrt(2) * z(accuracy) since it's a forced choice between two options.

    Args:
        trials: List of trial dicts with 'correct', 'response', 'target', 'foil_type' keys

    Returns:
        Dict with accuracy, d', mem_score, and per-foil-type breakdowns
    """
    if not trials:
        return {"accuracy": 0, "d_prime": 0, "mem_score": 0}

    n_correct = sum(1 for t in trials if t.get('correct', 0) == 1)
    n_total = len(trials)
    accuracy = n_correct / n_total if n_total > 0 else 0

    # Exclude parsing failures (response = -1) for valid accuracy
    valid_trials = [t for t in trials if t.get('response', -1) != -1]
    n_valid = len(valid_trials)
    n_valid_correct = sum(1 for t in valid_trials if t.get('correct', 0) == 1)
    valid_accuracy = n_valid_correct / n_valid if n_valid > 0 else 0

    # d' for 2-AFC: d' = sqrt(2) * z(accuracy)
    # Using adjusted accuracy to avoid infinity
    adj_acc = (n_valid_correct + 0.5) / (n_valid + 1) if n_valid > 0 else 0.5
    d_prime = np.sqrt(2) * norm.ppf(adj_acc)

    # Memory score: accuracy above chance (0.5), scaled to 0-1
    mem_score = max(0, min(1, 2 * (valid_accuracy - 0.5)))

    # Breakdown by foil type
    accuracy_by_type = {}
    d_prime_by_type = {}
    foil_types = set(t.get('foil_type', 'unknown') for t in trials)

    for ftype in foil_types:
        ftype_trials = [t for t in valid_trials if t.get('foil_type') == ftype]
        if ftype_trials:
            fc = sum(1 for t in ftype_trials if t.get('correct', 0) == 1)
            fn = len(ftype_trials)
            accuracy_by_type[ftype] = fc / fn
            adj = (fc + 0.5) / (fn + 1)
            d_prime_by_type[ftype] = float(np.sqrt(2) * norm.ppf(adj))

    return {
        "accuracy": float(accuracy),
        "valid_accuracy": float(valid_accuracy),
        "d_prime": float(d_prime),
        "mem_score": float(mem_score),
        "n_trials": n_total,
        "n_valid_trials": n_valid,
        "n_correct": n_correct,
        "n_parse_failures": n_total - n_valid,
        "accuracy_by_type": accuracy_by_type,
        "d_prime_by_type": d_prime_by_type,
    }


def calculate_mst_metrics(results):
    """
    Metrics for the Mnemonic Similarity Task.

    Each result dict must have:
      - type: 'target' | 'lure' | 'foil'
      - reported: model response ('old' | 'similar' | 'new')
      - metadata: dict with optional 'bin' (int 1-5) for lure items

    Returns hit_rate, false_alarm_rate, LDI (overall and per bin), and
    per-trial-type accuracy.
    """
    if not results:
        return {
            "accuracy": 0, "n_correct": 0, "total": 0,
            "hit_rate": 0, "false_alarm_rate": 0,
            "ldi": 0, "ldi_by_bin": {},
            "target_accuracy": 0, "lure_accuracy": 0, "foil_accuracy": 0,
            "p_similar_lure": 0, "p_similar_foil": 0,
        }

    def _norm(s):
        return str(s).strip().lower() if s is not None else ""

    targets = [r for r in results if r.get("type") == "target"]
    lures   = [r for r in results if r.get("type") == "lure"]
    foils   = [r for r in results if r.get("type") == "foil"]

    def _rate(items, response):
        if not items:
            return 0.0
        return float(np.mean([1 if _norm(r.get("reported")) == response else 0 for r in items]))

    hit_rate        = _rate(targets, "old")
    fa_rate         = _rate(foils, "old")
    p_sim_lure      = _rate(lures, "similar")
    p_sim_foil      = _rate(foils, "similar")
    ldi             = p_sim_lure - p_sim_foil

    # Target acc = p("old"|target), lure acc = p("similar"|lure), foil acc = p("new"|foil)
    target_accuracy = hit_rate
    lure_accuracy   = p_sim_lure
    foil_accuracy   = _rate(foils, "new")

    # LDI per similarity bin
    ldi_by_bin = {}
    for bin_num in range(1, 6):
        bin_lures = [r for r in lures if r.get("metadata", {}).get("bin") == bin_num]
        if bin_lures:
            ldi_by_bin[bin_num] = float(_rate(bin_lures, "similar") - p_sim_foil)

    n_correct = sum(1 for r in results if r.get("correct"))
    accuracy = n_correct / len(results) if results else 0.0

    return {
        "accuracy": float(accuracy),
        "n_correct": int(n_correct),
        "total": len(results),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(fa_rate),
        "p_similar_lure": float(p_sim_lure),
        "p_similar_foil": float(p_sim_foil),
        "ldi": float(ldi),
        "ldi_by_bin": ldi_by_bin,
        "target_accuracy": float(target_accuracy),
        "lure_accuracy": float(lure_accuracy),
        "foil_accuracy": float(foil_accuracy),
        "n_targets": len(targets),
        "n_lures": len(lures),
        "n_foils": len(foils),
    }


def calculate_associative_inference_metrics(results):
    """Calculates 2-AFC associative inference accuracy."""
    if not results:
        return {"accuracy": 0, "n_correct": 0, "total": 0}

    n_correct = 0
    total = len(results)

    for res in results:
        if res.get("target") == res.get("reported"):
            n_correct += 1

    return {
        "accuracy": n_correct / total if total > 0 else 0,
        "n_correct": n_correct,
        "total": total,
    }
