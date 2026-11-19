import math
import numpy as np


def compute_conformal_quantile(scores, alpha=0.10):
    """Finite-sample split-conformal quantile for independent patient-level scores."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        raise ValueError("Cannot calibrate conformal threshold with zero patient scores.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 1.0  # nonconformity scores lie in [0, 1]; conservative finite-sample endpoint
    return float(np.sort(values)[k - 1])


def patient_nonconformity_scores(probabilities, targets):
    """
    Return exact patient_max and exact ceil(0.95*K)-order-statistic patient_p95 scores over WT voxels.
    By using the exact order statistic sorted_scores[ceil(0.95*K) - 1] (no linear interpolation),
    the event {patient_p95 <= q} is algebraically equivalent to {WT voxel coverage >= 0.95},
    preserving exact finite-sample split-conformal coverage guarantees.
    """
    import torch

    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape (C, D, H, W)")
    if targets.ndim != 3 or tuple(probabilities.shape[1:]) != tuple(targets.shape):
        raise ValueError("target shape must match probability volume")
    probs_true = torch.gather(
        probabilities, dim=0, index=targets.long().unsqueeze(0)
    ).squeeze(0)
    wt = targets > 0
    if not wt.any():
        return None
    scores = (1.0 - probs_true[wt]).detach().float().cpu().numpy()
    sorted_scores = np.sort(scores)
    k_p95 = max(1, min(len(sorted_scores), math.ceil(0.95 * len(sorted_scores))))
    return {
        "patient_max": float(sorted_scores[-1]),
        "patient_p95": float(sorted_scores[k_p95 - 1]),
        "patient_mean": float(np.mean(scores)),
        "n_wt_voxels": int(len(scores)),
    }
