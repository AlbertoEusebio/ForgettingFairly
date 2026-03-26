# ─────────────────────────────────────────────────────────────
# kaggle_run.ipynb  –  paste this into a Kaggle notebook
#
# Datasets to add in the Kaggle UI before running:
#   1. skin-lesions  (HAM10000 metadata + images)
#      https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
#
# GPU: T4 x1  (free tier, ~25 min total)
# ─────────────────────────────────────────────────────────────

# ── Cell 1: clone your repo ───────────────────────────────────
# Replace with your actual repo URL once you push the code
"""
!git clone https://github.com/YOUR_USERNAME/forgetting-fairly.git
%cd forgetting-fairly
!pip install -r requirements.txt -q
"""

# ── Cell 2: download RadImageNet weights ──────────────────────
"""
# Option A: if you've uploaded the weights as a Kaggle dataset
import shutil
shutil.copy(
    "/kaggle/input/radimagenet-weights/RadImageNet_resnet50.pth",
    "./data/RadImageNet_resnet50.pth"
)

# Option B: download directly (requires internet toggle ON in Kaggle)
# !wget -q -O ./data/RadImageNet_resnet50.pth \
#   https://drive.google.com/uc?id=YOUR_GDRIVE_ID
"""

# ── Cell 3: symlink Kaggle data to expected paths ─────────────
"""
import os, shutil

os.makedirs("./data", exist_ok=True)

# HAM10000 metadata
shutil.copy(
    "/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/"
    "HAM10000_metadata.csv",
    "./data/HAM10000_metadata.csv"
)

# Images (Kaggle dataset has them split in two folders)
# We create a single flat directory for simplicity
os.makedirs("./data/HAM10000_images", exist_ok=True)

for part in ["HAM10000_images_part1", "HAM10000_images_part2"]:
    src = (f"/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/{part}")
    if os.path.exists(src):
        for fname in os.listdir(src):
            shutil.copy(os.path.join(src, fname),
                        f"./data/HAM10000_images/{fname}")

print("Data ready.")
"""

# ── Cell 4: quick sanity check before full run ────────────────
"""
import pandas as pd
df = pd.read_csv("./data/HAM10000_metadata.csv")
print(df.head())
print("\\nColumns:", df.columns.tolist())
print("\\nSources:", df["dataset"].value_counts())
print("\\nClasses:", df["dx"].value_counts())
print("\\nSex:",     df["sex"].value_counts())
"""

# ── Cell 5: run the pilot ─────────────────────────────────────
"""
!python pilot.py \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --img_dir ./data/HAM10000_images \
    --metadata ./data/HAM10000_metadata.csv \
    --weights ./data/RadImageNet_resnet50.pth
"""

# ── Cell 6: inspect results ───────────────────────────────────
"""
import json
with open("./results/pilot_results.json") as f:
    r = json.load(f)

print("After Task 1:")
print(f"  Overall AUC: {r['metrics_after_task1']['overall']['auc']:.4f}")
print(f"  EOD gap:     {r['metrics_after_task1']['eod_gap']:.4f}")

print("\\nAfter Task 2 (Task 1 val set):")
print(f"  Overall AUC: {r['metrics_after_task2']['overall']['auc']:.4f}")
print(f"  EOD gap:     {r['metrics_after_task2']['eod_gap']:.4f}")

delta_eod = r['metrics_after_task2']['eod_gap'] - r['metrics_after_task1']['eod_gap']
print(f"\\nΔ EOD gap:   {delta_eod:+.4f}  {'← hypothesis holds!' if delta_eod > 0 else ''}")
"""

# ── Cell 7: save output so it survives session end ────────────
"""
# Kaggle /kaggle/working/ persists as output after the session
import shutil, os
shutil.copy("./results/pilot_results.json",
            "/kaggle/working/pilot_results.json")
shutil.copy("./checkpoints/after_task1.pt",
            "/kaggle/working/after_task1.pt")
print("Saved to /kaggle/working/ ✓")
"""
