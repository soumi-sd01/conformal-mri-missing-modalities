import numpy as np
import pytest
import torch
from src.conformal.calibration import compute_conformal_quantile, patient_nonconformity_scores
from src.conformal.evaluation import prediction_set_from_probs, evaluate_patient_coverage


def test_conformal_quantile_uses_patient_scores():
    scores = np.linspace(0.01, 0.99, 100)
    q = compute_conformal_quantile(scores, alpha=0.10)
    assert 0.89 <= q <= 0.99


def test_empty_calibration_scores_rejected():
    with pytest.raises(ValueError):
        compute_conformal_quantile([], alpha=0.1)


def test_exact_p95_equivalence_without_interpolation_gap():
    # K=4 WT voxels: to achieve >= 0.95 coverage, all 4/4 voxels must be covered!
    # A linear-interpolated quantile would pick ~0.285 < 0.30 and fail coverage at q=p95.
    probs = torch.tensor([
        [[[0.1, 0.1], [0.1, 0.1]]],
        [[[0.9, 0.8], [0.8, 0.7]]],
        [[[0.0, 0.1], [0.1, 0.2]]],
        [[[0.0, 0.0], [0.0, 0.0]]],
    ]).reshape(4, 1, 2, 2)
    targets = torch.ones((1, 2, 2), dtype=torch.long)
    result = patient_nonconformity_scores(probs, targets)
    assert result["patient_max"] == pytest.approx(0.3)
    assert result["patient_p95"] == pytest.approx(0.3)
    ev_exact = evaluate_patient_coverage(prediction_set_from_probs(probs, result["patient_p95"]), targets)
    assert ev_exact["p95_patient_covered"] == 1.0
    ev_below = evaluate_patient_coverage(prediction_set_from_probs(probs, result["patient_p95"] - 1e-4), targets)
    assert ev_below["p95_patient_covered"] == 0.0


def test_prediction_set_and_patient_coverage():
    probs = torch.tensor([[[[0.9, 0.1]]], [[[0.1, 0.9]]]])
    targets = torch.tensor([[[0, 1]]])
    sets = prediction_set_from_probs(probs, 0.2)
    result = evaluate_patient_coverage(sets, targets)
    assert result["patient_covered"] == 1.0
    assert result["p95_patient_covered"] == 1.0
    assert result["voxel_coverage_WT"] == 1.0
