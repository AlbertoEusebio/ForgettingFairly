# ─────────────────────────────────────────────────────────────
# config.py  –  single source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────────────
# On Kaggle these will be under /kaggle/input/skin-lesions/
# Locally set via environment variable or edit directly
DATA_ROOT     = os.getenv("DATA_ROOT", "./data")
IMG_DIR       = os.path.join(DATA_ROOT, "HAM10000_images")
METADATA_PATH = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")

# RadImageNet weights – download from:
# https://github.com/BMEII-AI/RadImageNet  (ResNet50 .pth)
RADIMAGENET_WEIGHTS = os.getenv("RADIMAGENET_WEIGHTS", "./data/RadImageNet_resnet50.pth")

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

# ── Dataset ──────────────────────────────────────────────────
# Age-based domain split — both groups have mel/nv samples
TASK1_SOURCE  = "young"        # age < 50
TASK2_SOURCE  = "older"        # age >= 50
SOURCE_COLUMN = "_age_group"   # synthetic column, created in data.py

# Binary classification: melanoma (1) vs nevus (0)
CLASSES       = ["nv", "mel"]
SENSITIVE_COL = "sex"            # subgroup attribute

VAL_FRAC = 0.2
RANDOM_SEED = 42

# ── Model ────────────────────────────────────────────────────
FREEZE_BACKBONE = True   # frozen for cheap pilot; set False to experiment

# ── Training ─────────────────────────────────────────────────
EPOCHS     = 10          # enough for a frozen backbone linear probe
BATCH_SIZE = 64
LR         = 1e-3        # higher LR ok since only head is trained
WEIGHT_DECAY = 1e-4

# Class imbalance: melanoma is minority (~20% of nv+mel)
# Options: "weighted_loss" | "oversample" | None
IMBALANCE_STRATEGY = "weighted_loss"

# ── Evaluation ───────────────────────────────────────────────
# Threshold for converting prob → binary pred
DECISION_THRESHOLD = 0.5

# Minimum EOD gap considered "meaningful" for pilot go/no-go
EOD_GAP_THRESHOLD = 0.03
