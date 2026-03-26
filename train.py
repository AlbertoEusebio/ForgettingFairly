# ─────────────────────────────────────────────────────────────
# train.py  –  single-epoch training loop
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

import config


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
) -> dict:
    """
    Trains model for one full pass over loader.
    Returns a dict of training statistics for logging.
    """
    model.train()

    # ── Loss function ─────────────────────────────────────────
    # If the loader carries pre-computed class weights, use weighted CE.
    # This handles the mel/nv imbalance without oversampling cost.
    if loader.class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=loader.class_weights.to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct      = 0
    total        = 0

    for batch_idx, (imgs, labels, _subgroups) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # ── Batch statistics ──────────────────────────────────
        running_loss += loss.item() * imgs.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  epoch {epoch+1} | batch {batch_idx+1}/{len(loader)} | "
                f"loss {loss.item():.4f}"
            )

    stats = {
        "epoch":    epoch + 1,
        "loss":     running_loss / total,
        "acc":      correct / total,
    }
    print(f"[train] epoch {epoch+1} done – "
          f"loss: {stats['loss']:.4f}  acc: {stats['acc']:.4f}")
    return stats
