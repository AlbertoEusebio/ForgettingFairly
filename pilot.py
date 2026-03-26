# ─────────────────────────────────────────────────────────────
# pilot.py  –  entry point: Task 1 → checkpoint → Task 2 → compare
#
# Usage:
#   python pilot.py
#   python pilot.py --epochs 5 --batch_size 32
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import time

import torch

import config
from data import get_task_splits, make_loader
from evaluate import evaluate_fairness, print_delta_report
from model import get_model, get_trainable_params
from train import train_one_epoch


# ── CLI overrides (useful in Kaggle notebooks) ────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ForgettingFairly pilot")
    parser.add_argument("--epochs",     type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.LR)
    parser.add_argument("--img_dir",    type=str,   default=config.IMG_DIR)
    parser.add_argument("--metadata",   type=str,   default=config.METADATA_PATH)
    parser.add_argument("--weights",    type=str,   default=config.RADIMAGENET_WEIGHTS)
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply CLI overrides to config (keeps everything in one place)
    config.EPOCHS             = args.epochs
    config.BATCH_SIZE         = args.batch_size
    config.LR                 = args.lr
    config.IMG_DIR            = args.img_dir
    config.METADATA_PATH      = args.metadata
    config.RADIMAGENET_WEIGHTS = args.weights

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[pilot] Device: {device}")
    if device.type == "cuda":
        print(f"[pilot] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[pilot] VRAM: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────
    print("\n[pilot] Loading data splits...")
    t1_train_df, t1_val_df, t2_train_df, t2_val_df = get_task_splits()

    t1_train_loader = make_loader(t1_train_df, config.IMG_DIR, train=True)
    t1_val_loader   = make_loader(t1_val_df,   config.IMG_DIR, train=False)
    t2_train_loader = make_loader(t2_train_df, config.IMG_DIR, train=True)
    t2_val_loader   = make_loader(t2_val_df,   config.IMG_DIR, train=False)

    # ── Model ─────────────────────────────────────────────────
    print("\n[pilot] Building model...")
    model = get_model(num_classes=2).to(device)

    optimizer = torch.optim.Adam(
        get_trainable_params(model),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # ── TASK 1 ────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TASK 1: {config.TASK1_SOURCE}")
    print(f"{'─'*55}")

    t1_history = []
    t0 = time.time()
    for epoch in range(config.EPOCHS):
        stats = train_one_epoch(model, t1_train_loader, optimizer, device, epoch)
        t1_history.append(stats)

    print(f"\n[pilot] Task 1 training done in {time.time()-t0:.1f}s")

    # Evaluate on Task 1 val – this is the BEFORE snapshot
    print("\n[pilot] Evaluating on Task 1 val (before forgetting)...")
    metrics_before = evaluate_fairness(model, t1_val_loader, device,
                                       split_name="T1 val | after T1")

    # Save checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "after_task1.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics_before,
        "epoch": config.EPOCHS,
    }, ckpt_path)
    print(f"[pilot] Checkpoint saved → {ckpt_path}")

    # ── TASK 2 ────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TASK 2: {config.TASK2_SOURCE}  (plain fine-tuning, no CL)")
    print(f"{'─'*55}")

    # Reset optimizer state to simulate a new training phase
    optimizer = torch.optim.Adam(
        get_trainable_params(model),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    t2_history = []
    t0 = time.time()
    for epoch in range(config.EPOCHS):
        stats = train_one_epoch(model, t2_train_loader, optimizer, device, epoch)
        t2_history.append(stats)

    print(f"\n[pilot] Task 2 training done in {time.time()-t0:.1f}s")

    # Evaluate on Task 1 val again – this is the AFTER snapshot
    # Key: same val set as before; we're measuring forgetting on Task 1
    print("\n[pilot] Evaluating on Task 1 val (after forgetting)...")
    metrics_after = evaluate_fairness(model, t1_val_loader, device,
                                      split_name="T1 val | after T2")

    # Also evaluate on Task 2 val to confirm learning happened
    print("\n[pilot] Evaluating on Task 2 val (plasticity check)...")
    metrics_t2 = evaluate_fairness(model, t2_val_loader, device,
                                   split_name="T2 val | after T2")

    # ── Core result ───────────────────────────────────────────
    print_delta_report(metrics_before, metrics_after, task_label="Task 2")

    # ── Save results ──────────────────────────────────────────
    results = {
        "config": {
            "epochs":      config.EPOCHS,
            "batch_size":  config.BATCH_SIZE,
            "lr":          config.LR,
            "task1_source": config.TASK1_SOURCE,
            "task2_source": config.TASK2_SOURCE,
            "freeze_backbone": config.FREEZE_BACKBONE,
        },
        "metrics_after_task1":  metrics_before,
        "metrics_after_task2":  metrics_after,
        "metrics_task2_val":    metrics_t2,
        "training_history": {
            "task1": t1_history,
            "task2": t2_history,
        }
    }

    results_path = os.path.join(config.RESULTS_DIR, "pilot_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[pilot] Results saved → {results_path}")


if __name__ == "__main__":
    main()
