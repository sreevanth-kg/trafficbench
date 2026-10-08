"""Missing-aware speed metrics in the dataset's original mph units."""

import numpy as np


def regime_frequencies(data, origins):
    """Describe the fixed training-label regimes; never choose thresholds from test."""
    rows = []
    for split, values in origins.items():
        counts = np.zeros((data.horizon, 3), dtype=np.int64)
        for start in range(0, len(values), 256):
            batch = values[start:start + 256]
            truth = data.raw[batch[:, None] + np.arange(1, data.horizon + 1)]
            valid = np.isfinite(truth) & (truth > 0)
            changing = np.abs(truth - data.filled[batch, None]) >= 8
            congested = truth < 40
            masks = (valid & ~congested & ~changing, valid & congested, valid & ~congested & changing)
            counts += np.stack([mask.sum(axis=(0, 2)) for mask in masks], axis=-1)
        for step, counts_at_step in enumerate(counts):
            total = int(counts_at_step.sum())
            names = ("free_flow", "congested", "transition")
            rows.append({"split": split, "horizon_minutes": (step + 1) * 5, "valid_count": total,
                         "counts": dict(zip(names, counts_at_step.tolist())),
                         "fractions": {name: float(count / total) if total else None
                                       for name, count in zip(names, counts_at_step)}})
    return {"definition": "Congested: target < 40 mph; otherwise transition: absolute target change from latest causally filled input >= 8 mph; otherwise free_flow. Missing targets excluded; thresholds fixed across datasets.",
            "rows": rows}


def speed_metrics(prediction, truth, congestion_mph=40.0):
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth shapes differ.")
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions contain NaN or infinity.")
    valid = np.isfinite(truth) & (truth > 0)
    rows = []
    for minutes, step in ((15, 2), (30, 5), (60, 11)):
        if step >= prediction.shape[1]:
            raise ValueError("At least twelve five-minute forecast steps are required.")
        observed = truth[:, step]
        error = prediction[:, step].astype(np.float64) - observed
        for period, mask in (("all", valid[:, step]), ("congested", valid[:, step] & (observed < congestion_mph))):
            count = int(mask.sum())
            rows.append({
                "horizon_minutes": minutes, "period": period, "count": count,
                "mae_mph": float(np.abs(error[mask]).mean()) if count else None,
                "rmse_mph": float(np.sqrt(np.square(error[mask]).mean())) if count else None,
            })
    return rows
