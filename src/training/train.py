import argparse
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.unet3d import UNet3D
from src.training.losses import DiceCELoss
from src.training.metrics import region_dice
from src.training.utils import set_seed, make_generator, seed_worker
from src.data.training_dataset import BraTSPatchDataset


def apply_modality_dropout(images, probability=0.5, rng=None):
    """Per-sample channel dropout; at least one MRI sequence is always retained."""
    images = images.clone()
    batch_size, channels = images.shape[:2]
    r = rng if rng is not None else np.random
    for b in range(batch_size):
        if r.random() < probability:
            keep_count = int(r.randint(1, channels + 1)) if rng is None else int(r.integers(1, channels + 1))
            keep = r.choice(channels, size=keep_count, replace=False)
            drop = [i for i in range(channels) if i not in keep]
            if drop:
                images[b, drop] = 0.0
    return images


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    modality_dropout=False,
    dropout_probability=0.5,
    grad_clip=1.0,
    eval_dropout=False,
    seed=42,
):
    training = optimizer is not None
    model.train(training)
    losses, wt_scores = [], []
    val_rng = np.random.default_rng(seed + 777) if (not training and eval_dropout) else None
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if training and modality_dropout:
                images = apply_modality_dropout(images, dropout_probability)
            elif (not training) and eval_dropout:
                images = apply_modality_dropout(images, probability=1.0, rng=val_rng)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if not training:
                pred = logits.argmax(dim=1)
                wt_scores.append(region_dice(pred, masks)["WT"])
    return float(np.mean(losses)), (float(np.mean(wt_scores)) if wt_scores else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--modality-dropout", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints_v3")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-existing-best", action="store_true",
                        help="Skip training whenever best_model.pt already exists (protects early-stopped checkpoints).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["project"]["random_seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["training"].get("mixed_precision", True) and device.type == "cuda")

    tag = "modality_dropout" if args.modality_dropout else "baseline_complete"
    save_dir = Path(args.output_dir) / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"
    latest_path = save_dir / "last_checkpoint.pt"
    history_path = save_dir / "training_history.csv"
    max_epochs = int(args.epochs or cfg["training"]["max_epochs"])
    patience = int(cfg["training"].get("early_stopping_patience", 7))

    print(f"Device={device}; AMP={amp_enabled}; model={tag}")

    # Skip RAM caching & training if model finished max_epochs, early-stopped, or skip-existing-best is set.
    if not args.force_retrain:
        if args.skip_existing_best and best_path.exists():
            _best = torch.load(best_path, map_location="cpu", weights_only=False)
            print(
                f"Skipping {tag}: saved best_model.pt exists (epoch={_best.get('epoch', '?')}, "
                f"val_wt_dice={_best.get('val_wt_dice', 0):.4f}). Use --force-retrain to overwrite."
            )
            return
        if latest_path.exists():
            _ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
            _ep = int(_ckpt.get("epoch", 0))
            _stale = int(_ckpt.get("stale_epochs", 0))
            _done = bool(_ckpt.get("completed", False)) or (_ep >= max_epochs) or (_stale >= patience)
            if _done and best_path.exists():
                reason = "early-stopped" if _stale >= patience else f"{_ep}/{max_epochs} epochs"
                print(
                    f"Skipping {tag}: training already finished ({reason}, "
                    f"best_dice={_ckpt.get('best_dice', 0):.4f})."
                )
                return

    patch_size = tuple(cfg["preprocessing"]["patch_size"])
    data_root = cfg["data"]["root_dir"]
    patches_per_patient = cfg["training"].get("patches_per_patient", "auto")
    val_tumor_frac = float(cfg["training"].get("validation_tumor_fraction", 1.0))

    train_ds = BraTSPatchDataset(
        data_root,
        "data/splits/train.txt",
        patch_size,
        deterministic=False,
        seed=seed,
        tumor_fraction=cfg["training"].get("tumor_centered_fraction", 0.5),
        patches_per_patient=patches_per_patient,
    )
    # Tumor-centered validation patches avoid empty-mask Dice=1.0 inflation
    val_ds = BraTSPatchDataset(
        data_root,
        "data/splits/validation.txt",
        patch_size,
        deterministic=True,
        seed=seed + 100000,
        tumor_fraction=val_tumor_frac,
    )
    workers = min(int(cfg["training"].get("num_workers", 2)), 2)
    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        worker_init_fn=seed_worker,
    )

    model = UNet3D(
        in_channels=4,
        out_channels=4,
        base_filters=int(cfg["model"].get("base_filters", 16)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    criterion = DiceCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch, best_dice, stale = 1, -1.0, 0
    if not args.force_retrain and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt.get("best_dice", -1.0))
        stale = int(ckpt.get("stale_epochs", 0))
        print(f"Resuming from epoch {start_epoch}")

    max_seconds = float(cfg["training"].get("max_training_hours", 4.0)) * 3600
    started = time.time()
    rows = []
    if history_path.exists() and start_epoch > 1:
        import pandas as pd
        rows = pd.read_csv(history_path).to_dict("records")

    for epoch in range(start_epoch, max_epochs + 1):
        if time.time() - started >= max_seconds:
            print("Reached configured training time budget; stopping cleanly.")
            break
        epoch_start = time.time()
        train_loss, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            amp_enabled,
            args.modality_dropout,
            float(cfg["training"].get("modality_dropout_probability", 0.5)),
            float(cfg["training"].get("gradient_clip_norm", 1.0)),
        )
        val_loss, val_wt_full = run_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled)
        if args.modality_dropout:
            _, val_wt_drop = run_epoch(
                model, val_loader, criterion, device, amp_enabled=amp_enabled, eval_dropout=True, seed=seed
            )
            selection_metric = 0.5 * val_wt_full + 0.5 * val_wt_drop
        else:
            val_wt_drop = val_wt_full
            selection_metric = val_wt_full

        scheduler.step(selection_metric)
        improved = selection_metric > best_dice
        if improved:
            best_dice, stale = selection_metric, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_wt_dice": val_wt_full,
                    "val_wt_dice_drop": val_wt_drop,
                    "selection_metric": best_dice,
                    "epoch": epoch,
                    "config": cfg,
                },
                best_path,
            )
        else:
            stale += 1

        completed = (epoch >= max_epochs) or (stale >= patience)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
            "stale_epochs": stale,
            "completed": completed,
            "config": cfg,
        }
        torch.save(state, latest_path)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wt_dice": val_wt_full,
            "val_wt_dice_drop": val_wt_drop,
            "best_val_wt_dice": best_dice,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
        }
        rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(history_path, index=False)
        print(
            f"Epoch {epoch}/{max_epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_WT_full={val_wt_full:.4f} "
            f"| val_WT_drop={val_wt_drop:.4f} | best={best_dice:.4f} | {row['epoch_seconds']/60:.1f} min"
        )
        if stale >= patience:
            print(f"Early stopping after {patience} non-improving epochs.")
            break

    print(f"Best checkpoint: {best_path}")
    print(f"Latest resumable checkpoint: {latest_path}")
    print(f"Training history: {history_path}")
    print(f"Elapsed hours: {(time.time()-started)/3600:.2f}")


if __name__ == "__main__":
    main()
