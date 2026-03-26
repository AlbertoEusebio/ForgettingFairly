# ─────────────────────────────────────────────────────────────
# model.py  –  RadImageNet-pretrained ResNet50, frozen backbone,
#              trainable linear head
# ─────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torchvision.models as models

import config


def get_model(num_classes: int = 2) -> nn.Module:
    """
    Returns a ResNet50 with:
      - weights loaded from RadImageNet checkpoint (if available)
        else falls back to ImageNet weights with a clear warning
      - all layers frozen except the final FC head
      - FC head replaced with a fresh Linear(2048 → num_classes)

    Only the head parameters will appear in optimizer parameter groups.
    """
    # ── Build base architecture ───────────────────────────────
    model = models.resnet50(weights=None)  # no auto-download

    # ── Load RadImageNet weights ──────────────────────────────
    weights_path = config.RADIMAGENET_WEIGHTS
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")

        # RadImageNet checkpoint may be wrapped in a 'state_dict' key
        if "state_dict" in state:
            state = state["state_dict"]

        # Remap backbone.N.* keys → standard torchvision layer names,
        # strip 'module.' prefix, and drop the original FC head
        def _remap(state):
            PREFIX = {
                "backbone.0.": "conv1.",
                "backbone.1.": "bn1.",
                "backbone.4.": "layer1.",
                "backbone.5.": "layer2.",
                "backbone.6.": "layer3.",
                "backbone.7.": "layer4.",
            }
            out = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                for old, new in PREFIX.items():
                    if k.startswith(old):
                        k = new + k[len(old):]
                        break
                if not k.startswith("fc."):
                    out[k] = v
            return out

        state = _remap(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[model] Loaded RadImageNet weights from {weights_path}")
        if missing:
            print(f"[model] Missing keys (expected – these are the new head): "
                  f"{missing}")
        if unexpected:
            print(f"[model] Unexpected keys: {unexpected}")
    else:
        # Graceful fallback: ImageNet pretrained via torchvision
        print(
            f"[model] WARNING: RadImageNet weights not found at '{weights_path}'.\n"
            f"         Falling back to ImageNet pretrained weights.\n"
            f"         Download RadImageNet weights from:\n"
            f"         https://github.com/BMEII-AI/RadImageNet"
        )
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("[model] Loaded ImageNet pretrained weights.")
        except Exception:
            # Offline environment (e.g. restricted sandbox) — random init
            print("[model] WARNING: Could not download ImageNet weights either.\n"
                  "         Using random init. Results are for debugging only.")
            model = models.resnet50(weights=None)

    # ── Freeze backbone ───────────────────────────────────────
    if config.FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    # ── Replace classification head ───────────────────────────
    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc    = nn.Linear(in_features, num_classes)
    # Head is always trainable (requires_grad=True by default)

    # ── Report parameter counts ───────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parameters – total: {total:,}  trainable: {trainable:,}  "
          f"({100*trainable/total:.2f}%)")

    return model


def get_trainable_params(model: nn.Module):
    """Returns only the parameters that require gradients (i.e. the head)."""
    return [p for p in model.parameters() if p.requires_grad]
