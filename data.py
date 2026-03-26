# ─────────────────────────────────────────────────────────────
# data.py  –  HAM10000 loader with source-based task splits
#             and subgroup (sex) labelling
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

import config


# ── Transforms ───────────────────────────────────────────────

def get_transforms(train: bool) -> T.Compose:
    """
    Standard medical imaging augmentations.
    RadImageNet was trained on 224×224 with ImageNet normalisation stats,
    which are close enough to dermatoscopy for a frozen backbone.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── Dataset ───────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    Returns (image, label, subgroup) tuples.

    label:    0 = nevus (nv), 1 = melanoma (mel)
    subgroup: 0 = female, 1 = male
              -1 = unknown (excluded by default in get_task_splits)
    """

    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.transform = transform

        # Pre-compute numeric label and subgroup columns
        self.df["_label"]    = (self.df["dx"] == "mel").astype(int)
        self.df["_subgroup"] = self.df[config.SENSITIVE_COL].map(
            {"male": 1, "female": 0}
        ).fillna(-1).astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # HAM10000 images live in one or two folders; try both
        path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        if not os.path.exists(path):
            # Some Kaggle kernels unzip into subfolder _part1 / _part2
            for part in ["HAM10000_images_part1", "HAM10000_images_part2"]:
                candidate = os.path.join(
                    os.path.dirname(self.img_dir), part, f"{row['image_id']}.jpg"
                )
                if os.path.exists(candidate):
                    path = candidate
                    break

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, int(row["_label"]), int(row["_subgroup"])


# ── Splits ────────────────────────────────────────────────────

def get_task_splits(metadata_path: str = config.METADATA_PATH):
    """
    Loads HAM10000 metadata and returns four DataFrames:
        t1_train, t1_val  (ViDIR source  – Task 1)
        t2_train, t2_val  (Rosendahl src – Task 2)

    Only mel/nv classes are kept. Rows with unknown sex are dropped.
    """
    df = pd.read_csv(metadata_path)

    # ── Sanity-check column names ──────────────────────────────
    # The Kaggle HAM10000 CSV uses 'dataset' for the source column
    if "dataset" in df.columns and "datasource" not in df.columns:
        df = df.rename(columns={"dataset": "datasource"})

    required = {"image_id", "dx", "sex", "datasource"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── Filter ────────────────────────────────────────────────
    df = df[df["dx"].isin(config.CLASSES)].copy()
    df = df[df["sex"].isin(["male", "female"])].copy()  # drop unknowns

    print(f"[data] Total usable samples: {len(df)}")
    print(f"[data] Class distribution:\n{df['dx'].value_counts()}")
    print(f"[data] Sex distribution:\n{df['sex'].value_counts()}")
    print(f"[data] Source distribution:\n{df['datasource'].value_counts()}")

    # ── Task 1: ViDIR ─────────────────────────────────────────
    t1 = df[df["datasource"] == config.TASK1_SOURCE].copy()
    if len(t1) == 0:
        raise ValueError(
            f"No samples found for Task 1 source '{config.TASK1_SOURCE}'.\n"
            f"Available sources: {df['datasource'].unique().tolist()}"
        )

    t1_train, t1_val = train_test_split(
        t1, test_size=config.VAL_FRAC,
        stratify=t1["dx"], random_state=config.RANDOM_SEED
    )

    # ── Task 2: Rosendahl / QIMR ──────────────────────────────
    t2 = df[df["datasource"] == config.TASK2_SOURCE].copy()
    if len(t2) == 0:
        raise ValueError(
            f"No samples found for Task 2 source '{config.TASK2_SOURCE}'.\n"
            f"Available sources: {df['datasource'].unique().tolist()}"
        )

    t2_train, t2_val = train_test_split(
        t2, test_size=config.VAL_FRAC,
        stratify=t2["dx"], random_state=config.RANDOM_SEED
    )

    _print_split_stats("Task 1 train", t1_train)
    _print_split_stats("Task 1 val  ", t1_val)
    _print_split_stats("Task 2 train", t2_train)
    _print_split_stats("Task 2 val  ", t2_val)

    return t1_train, t1_val, t2_train, t2_val


def _print_split_stats(name: str, df: pd.DataFrame):
    print(f"\n[data] {name}: n={len(df)}")
    print(f"       mel={( df['dx']=='mel').sum()}  "
          f"nv={(df['dx']=='nv').sum()}")
    print(f"       male={(df['sex']=='male').sum()}  "
          f"female={(df['sex']=='female').sum()}")


# ── DataLoaders ───────────────────────────────────────────────

def make_loader(df: pd.DataFrame, img_dir: str,
                train: bool, batch_size: int = config.BATCH_SIZE,
                num_workers: int = 2) -> DataLoader:
    """
    Builds a DataLoader. If train=True and IMBALANCE_STRATEGY is
    'weighted_loss', returns class weights instead of a sampler
    (weights are attached to the loader as .class_weights attribute).
    """
    dataset = HAM10000Dataset(df, img_dir, transform=get_transforms(train))

    sampler = None
    if train and config.IMBALANCE_STRATEGY == "oversample":
        labels  = (df["dx"] == "mel").astype(int).values
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Attach class weights for weighted CE loss (computed from training df)
    if train and config.IMBALANCE_STRATEGY == "weighted_loss":
        labels  = (df["dx"] == "mel").astype(int).values
        counts  = np.bincount(labels, minlength=2)
        w       = len(labels) / (2.0 * counts)         # inverse-frequency
        loader.class_weights = torch.tensor(w, dtype=torch.float32)
    else:
        loader.class_weights = None

    return loader
