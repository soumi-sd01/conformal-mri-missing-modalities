import math
import numpy as np
import torch


def prediction_set_from_probs(probabilities, threshold):
    return (1.0 - probabilities) <= float(threshold)


def evaluate_patient_coverage(prediction_sets, targets, p95_tol=0.95):
    """Simultaneous WT coverage (patient_max) plus exact >=95% risk-controlled coverage & set size."""
    true_in_set = torch.gather(
        prediction_sets, dim=0, index=targets.long().unsqueeze(0)
    ).squeeze(0)
    wt = targets > 0
    if not wt.any():
        return {
            "patient_covered": np.nan,
            "p95_patient_covered": np.nan,
            "voxel_coverage_WT": np.nan,
            "mean_set_size_WT": np.nan,
        }
    covered = true_in_set[wt]
    n_wt = int(covered.numel())
    n_cov = int(covered.sum().item())
    voxel_cov = float(n_cov / n_wt)
    req_cov = math.ceil(p95_tol * n_wt)
    set_size = prediction_sets.sum(dim=0)[wt].float()
    return {
        "patient_covered": float(n_cov == n_wt),
        "p95_patient_covered": float(n_cov >= req_cov),
        "voxel_coverage_WT": voxel_cov,
        "mean_set_size_WT": float(set_size.mean().item()),
    }


def compute_failure_detection_diagnostics(uncertainties, dice_scores):
    u = np.asarray(uncertainties, dtype=float)
    err = 1.0 - np.asarray(dice_scores, dtype=float)
    mask = np.isfinite(u) & np.isfinite(err)
    u, err = u[mask], err[mask]
    if len(u) < 3 or np.std(u) < 1e-8 or np.std(err) < 1e-8:
        return {"error_correlation_r": float("nan"), "error_correlation_p": float("nan")}
    try:
        from scipy.stats import pearsonr
        res = pearsonr(u, err)
        return {"error_correlation_r": float(res.statistic), "error_correlation_p": float(res.pvalue)}
    except Exception:
        r = float(np.corrcoef(u, err)[0, 1])
        return {"error_correlation_r": r, "error_correlation_p": float("nan")}
