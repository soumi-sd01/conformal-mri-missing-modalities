import argparse
import hashlib
import json
import time
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.models.unet3d import UNet3D
from src.data.dataset import BraTSDataset
from src.training.metrics import region_dice
from src.conformal.calibration import compute_conformal_quantile, patient_nonconformity_scores
from src.conformal.evaluation import (
    prediction_set_from_probs,
    evaluate_patient_coverage,
    compute_failure_detection_diagnostics,
)

MODALITIES = ["flair", "t1", "t1ce", "t2"]


def get_patterns():
    """All 15 non-empty availability subsets, represented by missing channels."""
    patterns = {"full_4_modalities": []}
    for m in MODALITIES:
        patterns[f"missing_{m}"] = [m]
    for pair in combinations(MODALITIES, 2):
        patterns[f"missing_{'_'.join(pair)}"] = list(pair)
    for m in MODALITIES:
        patterns[f"only_{m}"] = [x for x in MODALITIES if x != m]
    assert len(patterns) == 15
    return patterns


def _starts(length, patch, stride):
    if length <= patch:
        return [0]
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def compute_file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def predict_volume_sliding_window(
    model,
    image,
    device,
    patch_size=(64, 64, 64),
    overlap=0.25,
    amp=True,
    batch_size=32,
):
    """Batched overlapping sliding-window logits; accumulators remain on CPU to limit VRAM."""
    if image.ndim != 4:
        raise ValueError("image must have shape (4, D, H, W)")
    original = tuple(image.shape[1:])
    pd_sz, ph_sz, pw_sz = patch_size
    pad_d = max(0, pd_sz - original[0])
    pad_h = max(0, ph_sz - original[1])
    pad_w = max(0, pw_sz - original[2])
    if pad_d or pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))

    d, h, w = image.shape[1:]
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)
    ds, hs, ws = (_starts(n, p, s) for n, p, s in zip((d, h, w), patch_size, stride))
    coords = [(z, y, x) for z in ds for y in hs for x in ws]

    sums = torch.zeros((4, d, h, w), dtype=torch.float32, device="cpu")
    counts = torch.zeros((1, d, h, w), dtype=torch.float32, device="cpu")

    bs = max(1, int(batch_size))
    idx = 0
    while idx < len(coords):
        chunk_coords = coords[idx : idx + bs]
        batch_patches = torch.stack(
            [image[:, z : z + pd_sz, y : y + ph_sz, x : x + pw_sz] for z, y, x in chunk_coords],
            dim=0,
        )
        try:
            batch_gpu = batch_patches.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                logits_batch = model(batch_gpu)
            logits_cpu = logits_batch.float().cpu()
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            continue

        for k, (z, y, x) in enumerate(chunk_coords):
            sums[:, z : z + pd_sz, y : y + ph_sz, x : x + pw_sz] += logits_cpu[k]
            counts[:, z : z + pd_sz, y : y + ph_sz, x : x + pw_sz] += 1.0
        idx += len(chunk_coords)

    result = sums / counts.clamp_min(1.0)
    return result[:, : original[0], : original[1], : original[2]]


def mask_missing(image, missing):
    result = image.clone()
    for modality in missing:
        result[MODALITIES.index(modality)] = 0.0
    return result


def save_json_atomic(path, payload):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--max-calibration-patients",
        type=int,
        default=0,
        help="0 means all calibration patients; use a small number only for smoke tests.",
    )
    parser.add_argument(
        "--max-test-patients",
        type=int,
        default=0,
        help="0 means all test patients; use a small number only for smoke tests.",
    )
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--cache-flush-every", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["inference"].get("use_mixed_precision", True))
    patch_size = tuple(cfg["inference"]["sliding_window_size"])
    overlap = float(args.overlap if args.overlap is not None else cfg["inference"]["overlap"])
    patch_bs = int(
        args.patch_batch_size
        if args.patch_batch_size is not None
        else cfg["inference"].get("patch_batch_size", 32)
    )
    flush_every = int(
        args.cache_flush_every
        if args.cache_flush_every is not None
        else cfg.get("evaluation", {}).get("cache_flush_every_n_patients", 10)
    )

    ckpt_sha256 = compute_file_sha256(args.checkpoint)
    model = UNet3D(4, 4, int(cfg["model"].get("base_filters", 16))).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_root = cfg["data"]["root_dir"]
    cal_ds = BraTSDataset(data_root, "data/splits/calibration.txt")
    test_ds = BraTSDataset(data_root, "data/splits/test.txt")
    cal_count = len(cal_ds) if args.max_calibration_patients <= 0 else min(len(cal_ds), args.max_calibration_patients)
    test_count = len(test_ds) if args.max_test_patients <= 0 else min(len(test_ds), args.max_test_patients)
    patterns = get_patterns()
    pattern_names = list(patterns)
    alpha = float(cfg["conformal"]["alpha"])

    print(
        f"Device={device} | patch_batch_size={patch_bs} | AMP={amp} | "
        f"cal={cal_count}/{len(cal_ds)} | test={test_count}/{len(test_ds)}"
    )
    print(f"Checkpoint SHA256={ckpt_sha256[:12]}... | overlap={overlap} | alpha={alpha}")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_meta = {
        "pipeline_version": "v5.1",
        "checkpoint_sha256": ckpt_sha256,
        "overlap": overlap,
        "patch_size": list(patch_size),
        "alpha": alpha,
        "calibration_patients": cal_count,
        "test_patients": test_count,
        "patterns": pattern_names,
    }

    # --- Phase 1: Calibration ---
    cal_cache_path = out_csv.with_name(out_csv.stem + "_calibration_patient_scores.json")
    cal_cache = {"meta": cache_meta, "patients": {}}
    if cal_cache_path.exists():
        try:
            previous = json.loads(cal_cache_path.read_text())
            if previous.get("meta") == cache_meta:
                cal_cache = previous
                print(f"Resuming cached calibration patients: {len(cal_cache['patients'])}/{cal_count}")
            else:
                print("Calibration cache mismatch (checkpoint hash or config changed); starting fresh pass.")
        except Exception:
            print("Calibration cache unreadable; starting fresh calibration pass.")

    cal_start = time.time()
    processed_new_cal = 0
    for i, patient_id in enumerate(cal_ds.patient_ids[:cal_count]):
        cached = cal_cache["patients"].get(patient_id, {})
        if cached.get("_skip") or all(name in cached for name in pattern_names):
            continue
        sample = cal_ds[i]
        image, target = sample["image"], sample["mask"]
        if not torch.any(target > 0):
            cal_cache["patients"][patient_id] = {"_skip": True}
            processed_new_cal += 1
            continue
        this_patient = {}
        for name, missing in patterns.items():
            masked = mask_missing(image, missing)
            logits = predict_volume_sliding_window(
                model, masked, device, patch_size, overlap, amp, batch_size=patch_bs
            )
            probs = torch.softmax(logits.float(), dim=0)
            score = patient_nonconformity_scores(probs, target)
            if score is not None:
                this_patient[name] = {
                    "patient_max": score["patient_max"],
                    "patient_p95": score["patient_p95"],
                    "patient_mean": score["patient_mean"],
                    "n_wt_voxels": score["n_wt_voxels"],
                }
        cal_cache["patients"][patient_id] = this_patient
        processed_new_cal += 1
        if processed_new_cal % flush_every == 0 or i + 1 == cal_count:
            save_json_atomic(cal_cache_path, cal_cache)
            elapsed = time.time() - cal_start
            rate = elapsed / max(1, processed_new_cal)
            remaining = max(0, cal_count - (i + 1))
            print(
                f"Calibration {i+1}/{cal_count} ({rate:.1f}s/pt | ETA {(remaining*rate)/60:.1f} min)"
            )
    save_json_atomic(cal_cache_path, cal_cache)

    patient_pattern_max = {name: [] for name in pattern_names}
    patient_pattern_p95 = {name: [] for name in pattern_names}
    patient_full_max, patient_global_max = [], []
    patient_full_p95, patient_global_p95 = [], []

    for patient_id in cal_ds.patient_ids[:cal_count]:
        ps = cal_cache["patients"].get(patient_id, {})
        if ps.get("_skip") or not all(name in ps for name in pattern_names):
            continue
        for name in pattern_names:
            patient_pattern_max[name].append(ps[name]["patient_max"])
            patient_pattern_p95[name].append(ps[name]["patient_p95"])
        patient_full_max.append(ps["full_4_modalities"]["patient_max"])
        patient_global_max.append(max(ps[name]["patient_max"] for name in pattern_names))
        patient_full_p95.append(ps["full_4_modalities"]["patient_p95"])
        patient_global_p95.append(max(ps[name]["patient_p95"] for name in pattern_names))

    if not patient_full_max or not patient_global_max:
        raise RuntimeError("Insufficient complete calibration patients; check dataset and inference.")

    min_group_n = int(cfg["conformal"].get("min_group_patients", 20))
    if cal_count >= min_group_n:
        small_groups = {
            name: len(scores) for name, scores in patient_pattern_max.items() if len(scores) < min_group_n
        }
        if small_groups:
            raise RuntimeError(f"Groups below min_group_patients ({min_group_n}): {small_groups}")

    q_a = compute_conformal_quantile(patient_full_max, alpha)
    q_b = {name: compute_conformal_quantile(scores, alpha) for name, scores in patient_pattern_max.items()}
    q_c = compute_conformal_quantile(patient_global_max, alpha)

    q_a_p95 = compute_conformal_quantile(patient_full_p95, alpha)
    q_b_p95 = {name: compute_conformal_quantile(scores, alpha) for name, scores in patient_pattern_p95.items()}
    q_c_p95 = compute_conformal_quantile(patient_global_p95, alpha)

    cal_summary = {
        "alpha": alpha,
        "checkpoint_sha256": ckpt_sha256,
        "calibration_patients_used": len(patient_global_max),
        "strategy_A_full_modality_q": q_a,
        "strategy_B_pattern_q": q_b,
        "strategy_C_all_pattern_patient_max_q": q_c,
        "p95_strategy_A_q": q_a_p95,
        "p95_strategy_B_pattern_q": q_b_p95,
        "p95_strategy_C_global_q": q_c_p95,
        "patient_level_calibration_scores": True,
        "exact_p95_order_statistic": True,
        "voxel_independence_assumed": False,
        "pattern_calibration_counts": {k: len(v) for k, v in patient_pattern_max.items()},
        "mean_patient_p95_diagnostic": {k: float(np.mean(v)) for k, v in patient_pattern_p95.items()},
    }
    calibration_json = out_csv.with_name(out_csv.stem + "_calibration.json")
    calibration_json.write_text(json.dumps(cal_summary, indent=2), encoding="utf-8")
    print(f"Saved calibration thresholds: {calibration_json}")

    # --- Phase 2: Held-out Test Evaluation ---
    test_cache_path = out_csv.with_name(out_csv.stem + "_test_patient_metrics.json")
    test_cache = {"meta": cache_meta, "patients": {}}
    if test_cache_path.exists():
        try:
            previous = json.loads(test_cache_path.read_text())
            if previous.get("meta") == cache_meta:
                test_cache = previous
                print(f"Resuming cached test patients: {len(test_cache['patients'])}/{test_count}")
            else:
                print("Test cache mismatch (checkpoint hash or config changed); starting fresh pass.")
        except Exception:
            print("Test cache unreadable; starting fresh test pass.")

    test_start = time.time()
    processed_new_test = 0
    for j, patient_id in enumerate(test_ds.patient_ids[:test_count]):
        cached = test_cache["patients"].get(patient_id, {})
        if cached.get("_skip") or all(name in cached for name in pattern_names):
            continue
        sample = test_ds[j]
        image, target = sample["image"], sample["mask"]
        if not torch.any(target > 0):
            test_cache["patients"][patient_id] = {"_skip": True}
            processed_new_test += 1
            continue
        patient_results = {}
        for name, missing in patterns.items():
            logits = predict_volume_sliding_window(
                model, mask_missing(image, missing), device, patch_size, overlap, amp, batch_size=patch_bs
            )
            probs = torch.softmax(logits.float(), dim=0)
            pred = logits.argmax(dim=0)
            dice = region_dice(pred.unsqueeze(0), target.unsqueeze(0))

            a_cov = evaluate_patient_coverage(prediction_set_from_probs(probs, q_a), target)
            b_cov = evaluate_patient_coverage(prediction_set_from_probs(probs, q_b[name]), target)
            c_cov = evaluate_patient_coverage(prediction_set_from_probs(probs, q_c), target)

            a_p95 = evaluate_patient_coverage(prediction_set_from_probs(probs, q_a_p95), target)
            b_p95 = evaluate_patient_coverage(prediction_set_from_probs(probs, q_b_p95[name]), target)
            c_p95 = evaluate_patient_coverage(prediction_set_from_probs(probs, q_c_p95), target)

            patient_results[name] = {
                "dice_wt": dice["WT"],
                "dice_tc": dice["TC"],
                "dice_et": dice["ET"],
                "cov_a": a_cov["patient_covered"],
                "cov_b": b_cov["patient_covered"],
                "cov_c": c_cov["patient_covered"],
                "voxel_b": b_cov["voxel_coverage_WT"],
                "set_size_a": a_cov["mean_set_size_WT"],
                "set_size_b": b_cov["mean_set_size_WT"],
                "set_size_c": c_cov["mean_set_size_WT"],
                "p95_cov_a": a_p95["p95_patient_covered"],
                "p95_cov_b": b_p95["p95_patient_covered"],
                "p95_cov_c": c_p95["p95_patient_covered"],
                "p95_voxel_b": b_p95["voxel_coverage_WT"],
                "p95_set_size_a": a_p95["mean_set_size_WT"],
                "p95_set_size_b": b_p95["mean_set_size_WT"],
                "p95_set_size_c": c_p95["mean_set_size_WT"],
            }
        test_cache["patients"][patient_id] = patient_results
        processed_new_test += 1
        if processed_new_test % flush_every == 0 or j + 1 == test_count:
            save_json_atomic(test_cache_path, test_cache)
            elapsed = time.time() - test_start
            rate = elapsed / max(1, processed_new_test)
            remaining = max(0, test_count - (j + 1))
            print(
                f"Test {j+1}/{test_count} ({rate:.1f}s/pt | ETA {(remaining*rate)/60:.1f} min)"
            )
    save_json_atomic(test_cache_path, test_cache)

    patient_rows = []
    stats = {
        name: {
            k: []
            for k in [
                "dice_wt", "dice_tc", "dice_et",
                "cov_a", "cov_b", "cov_c", "voxel_b", "set_size_a", "set_size_b", "set_size_c",
                "p95_cov_a", "p95_cov_b", "p95_cov_c", "p95_voxel_b", "p95_set_size_a", "p95_set_size_b", "p95_set_size_c",
            ]
        }
        for name in pattern_names
    }
    for patient_id in test_ds.patient_ids[:test_count]:
        pr = test_cache["patients"].get(patient_id, {})
        if pr.get("_skip"):
            continue
        for name in pattern_names:
            if name not in pr:
                continue
            r = pr[name]
            patient_rows.append({"patient_id": patient_id, "modality_pattern": name, **r})
            for k in stats[name]:
                stats[name][k].append(r[k])

    patient_csv = out_csv.with_name(out_csv.stem + "_patient_level.csv")
    pd.DataFrame(patient_rows).to_csv(patient_csv, index=False)

    records = []
    for pattern_index, name in enumerate(pattern_names, start=1):
        s = stats[name]
        dice_wt = s["dice_wt"]
        diag_max = compute_failure_detection_diagnostics(s["set_size_b"], dice_wt)
        diag_p95 = compute_failure_detection_diagnostics(s["p95_set_size_b"], dice_wt)
        records.append({
            "Modality_Pattern": name,
            "Missing_Count": len(patterns[name]),
            "Patients_Evaluated": len(dice_wt),
            "Dice_WT_Mean": float(np.nanmean(dice_wt)) if dice_wt else np.nan,
            "Dice_TC_Mean": float(np.nanmean(s["dice_tc"])) if s["dice_tc"] else np.nan,
            "Dice_ET_Mean": float(np.nanmean(s["dice_et"])) if s["dice_et"] else np.nan,
            "Strategy_A_Full_Cal_Coverage_WT": float(np.nanmean(s["cov_a"])) if s["cov_a"] else np.nan,
            "Strategy_B_Conditional_Coverage_WT": float(np.nanmean(s["cov_b"])) if s["cov_b"] else np.nan,
            "Strategy_C_Joint_Global_Coverage_WT": float(np.nanmean(s["cov_c"])) if s["cov_c"] else np.nan,
            "Global_Coverage_WT": float(np.nanmean(s["cov_c"])) if s["cov_c"] else np.nan,
            "Cond_Coverage_WT": float(np.nanmean(s["cov_b"])) if s["cov_b"] else np.nan,
            "Voxel_Coverage_WT_Strategy_B": float(np.nanmean(s["voxel_b"])) if s["voxel_b"] else np.nan,
            "Mean_Set_Size_Strategy_A": float(np.nanmean(s["set_size_a"])) if s["set_size_a"] else np.nan,
            "Cond_Mean_Set_Size": float(np.nanmean(s["set_size_b"])) if s["set_size_b"] else np.nan,
            "Mean_Set_Size_Strategy_C": float(np.nanmean(s["set_size_c"])) if s["set_size_c"] else np.nan,
            "Threshold_A": q_a,
            "Threshold_B": q_b[name],
            "Threshold_C": q_c,
            "Failure_Corr_r": diag_max["error_correlation_r"],
            "Failure_Corr_p": diag_max["error_correlation_p"],
            "P95_Strategy_A_Coverage_WT": float(np.nanmean(s["p95_cov_a"])) if s["p95_cov_a"] else np.nan,
            "P95_Strategy_B_Coverage_WT": float(np.nanmean(s["p95_cov_b"])) if s["p95_cov_b"] else np.nan,
            "P95_Strategy_C_Coverage_WT": float(np.nanmean(s["p95_cov_c"])) if s["p95_cov_c"] else np.nan,
            "P95_Voxel_Coverage_WT_Strategy_B": float(np.nanmean(s["p95_voxel_b"])) if s["p95_voxel_b"] else np.nan,
            "P95_Mean_Set_Size_Strategy_A": float(np.nanmean(s["p95_set_size_a"])) if s["p95_set_size_a"] else np.nan,
            "P95_Cond_Mean_Set_Size": float(np.nanmean(s["p95_set_size_b"])) if s["p95_set_size_b"] else np.nan,
            "P95_Mean_Set_Size_Strategy_C": float(np.nanmean(s["p95_set_size_c"])) if s["p95_set_size_c"] else np.nan,
            "P95_Threshold_A": q_a_p95,
            "P95_Threshold_B": q_b_p95[name],
            "P95_Threshold_C": q_c_p95,
            "P95_Failure_Corr_r": diag_p95["error_correlation_r"],
            "P95_Failure_Corr_p": diag_p95["error_correlation_p"],
            "Calibration_Patients": len(patient_pattern_max[name]),
        })
        print(
            f"[{pattern_index:02d}/15 {name:18s}] WT_Dice={np.nanmean(dice_wt):.3f} | "
            f"Max(A/B/C)={np.nanmean(s['cov_a'])*100:.1f}/{np.nanmean(s['cov_b'])*100:.1f}/{np.nanmean(s['cov_c'])*100:.1f}% | "
            f"P95(A/B/C)={np.nanmean(s['p95_cov_a'])*100:.1f}/{np.nanmean(s['p95_cov_b'])*100:.1f}/{np.nanmean(s['p95_cov_c'])*100:.1f}% "
            f"(sz={np.nanmean(s['p95_set_size_b']):.2f})"
        )

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved evaluation matrix: {out_csv}")
    print(f"Saved per-patient metrics: {patient_csv}")


if __name__ == "__main__":
    main()
