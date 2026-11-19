from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)


def plot_subgroup_coverage_comparison(
    df, target_coverage=0.90, save_path="figures/coverage.png", mode="max"
):
    labels = df["Modality_Pattern"].astype(str).tolist()
    if mode == "p95" and "P95_Strategy_A_Coverage_WT" in df.columns:
        a = df["P95_Strategy_A_Coverage_WT"].to_numpy(dtype=float)
        b = df["P95_Strategy_B_Coverage_WT"].to_numpy(dtype=float)
        c = df["P95_Strategy_C_Coverage_WT"].to_numpy(dtype=float)
        ylabel = "Patient-level P95 WT coverage (>=95% voxels)"
        title = "Risk-controlled conformal coverage (exact P95) by missing-modality pattern"
    else:
        a = df["Strategy_A_Full_Cal_Coverage_WT"].to_numpy(dtype=float)
        b = df["Strategy_B_Conditional_Coverage_WT"].to_numpy(dtype=float)
        c = df["Strategy_C_Joint_Global_Coverage_WT"].to_numpy(dtype=float)
        ylabel = "Patient-level simultaneous WT coverage"
        title = "Simultaneous conformal coverage (patient_max) by missing-modality pattern"

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x - width, a, width, color="#94A3B8", label="Strategy A: Full-modality q")
    ax.bar(x, b, width, color="#0284C7", label="Strategy B: Pattern-conditional q_m")
    ax.bar(x + width, c, width, color="#0F766E", label="Strategy C: Joint global q_global")
    ax.axhline(target_coverage, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Target {target_coverage:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    _clean_axes(ax)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_set_size_and_dice(df, save_path="figures/set_size_vs_dice.png"):
    x = df["Missing_Count"].to_numpy(dtype=float)
    sz_col = "P95_Cond_Mean_Set_Size" if "P95_Cond_Mean_Set_Size" in df.columns else "Cond_Mean_Set_Size"
    y = df[sz_col].to_numpy(dtype=float)
    d = df["Dice_WT_Mean"].to_numpy(dtype=float)

    grouped = df.groupby("Missing_Count", as_index=False).agg(
        mean_dice=("Dice_WT_Mean", "mean"),
        mean_sz=(sz_col, "mean"),
    )

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.scatter(x - 0.05, y, color="#0284C7", alpha=0.65, s=48, label="Pattern mean set size")
    ax1.plot(grouped["Missing_Count"], grouped["mean_sz"], color="#0284C7", linewidth=2, label="Mean set size trend")
    ax1.set_xlabel("Number of missing MRI modalities")
    ax1.set_ylabel("Mean conformal prediction-set size (WT)", color="#0284C7")
    ax1.set_xticks([0, 1, 2, 3])
    _clean_axes(ax1)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.scatter(x + 0.05, d, color="#D97706", alpha=0.65, marker="s", s=45, label="Pattern WT Dice")
    ax2.plot(grouped["Missing_Count"], grouped["mean_dice"], color="#D97706", linestyle="--", linewidth=2, label="WT Dice trend")
    ax2.set_ylabel("Mean Whole-Tumor (WT) Dice", color="#D97706")
    ax2.set_ylim(0, 1.0)
    ax1.set_title("Uncertainty expansion vs. segmentation degradation across missing modalities", fontweight="bold")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(df_base, df_drop, save_path="figures/model_comparison.png"):
    merged = df_base.merge(df_drop, on="Modality_Pattern", suffixes=("_base", "_drop"))
    labels = merged["Modality_Pattern"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].bar(x - width / 2, merged["Dice_WT_Mean_base"], width, color="#64748B", label="Baseline (Complete-Modality Trained)")
    axes[0].bar(x + width / 2, merged["Dice_WT_Mean_drop"], width, color="#0F766E", label="Modality-Dropout Trained")
    axes[0].set_ylabel("Mean WT Dice")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Baseline vs. Modality-Dropout across all 15 availability patterns", fontweight="bold")
    _clean_axes(axes[0])
    axes[0].legend(frameon=False)

    cov_col = "P95_Strategy_B_Coverage_WT" if "P95_Strategy_B_Coverage_WT_base" in merged.columns else "Strategy_B_Conditional_Coverage_WT"
    axes[1].bar(x - width / 2, merged[f"{cov_col}_base"], width, color="#64748B", label="Baseline Strategy B coverage")
    axes[1].bar(x + width / 2, merged[f"{cov_col}_drop"], width, color="#0284C7", label="Dropout Strategy B coverage")
    axes[1].axhline(0.90, color="#DC2626", linestyle="--", linewidth=1.4, label="Target 90%")
    axes[1].set_ylabel("Strategy B coverage")
    axes[1].set_ylim(0, 1.06)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
    _clean_axes(axes[1])
    axes[1].legend(frameon=False)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_curves(base_hist_path, drop_hist_path, save_path="figures/training_curves.png"):
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, path, color in [
        ("Baseline (Complete)", Path(base_hist_path), "#64748B"),
        ("Modality Dropout", Path(drop_hist_path), "#0F766E"),
    ]:
        if not path.exists():
            continue
        h = pd.read_csv(path)
        axes[0].plot(h["epoch"], h["train_loss"], linestyle="--", color=color, alpha=0.6, label=f"{label} train")
        axes[0].plot(h["epoch"], h["val_loss"], color=color, linewidth=2, label=f"{label} val")
        axes[1].plot(h["epoch"], h["val_wt_dice"], color=color, linewidth=2, marker="o", markersize=3, label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("DiceCE Loss")
    axes[0].set_title("Training & Validation Loss")
    _clean_axes(axes[0])
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation WT Dice")
    axes[1].set_title("Validation Whole-Tumor Dice Trajectory")
    axes[1].set_ylim(0.4, 0.95)
    _clean_axes(axes[1])
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
