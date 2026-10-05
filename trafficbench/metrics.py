"""Missing-aware speed metrics in the dataset's original mph units."""

import numpy as np


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
