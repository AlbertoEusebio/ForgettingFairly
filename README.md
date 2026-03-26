# ForgettingFairly — Pilot

**Hypothesis:** After continual learning on a new data source (Task 2),
a model forgets Task 1 *unevenly across demographic subgroups* — some
groups are forgotten faster than others.

## Project structure

```
forgetting_fairly/
├── config.py       # all hyperparameters in one place
├── data.py         # HAM10000 loader, task splits, subgroup labels
├── model.py        # RadImageNet ResNet50, frozen backbone
├── train.py        # single-epoch training loop
├── evaluate.py     # per-subgroup fairness metrics + forgetting report
├── pilot.py        # main script: Task 1 → checkpoint → Task 2 → compare
├── kaggle_run.py   # step-by-step Kaggle notebook cells
└── requirements.txt
```

## Pilot design

| | |
|---|---|
| **Dataset** | HAM10000 (skin lesion classification) |
| **Task 1** | ViDIR Vienna source (`vidir_modern`) |
| **Task 2** | Rosendahl/QIMR source (`rosendahl`) |
| **Classes** | melanoma (1) vs nevus (0) |
| **Subgroup** | sex (male / female) |
| **CL method** | Plain fine-tuning (no regularization — worst case baseline) |
| **Backbone** | ResNet50 pretrained on RadImageNet, **frozen** |

## Quick start (local)

```bash
pip install -r requirements.txt

# Set paths
export DATA_ROOT=./data
export RADIMAGENET_WEIGHTS=./data/RadImageNet_resnet50.pth

python pilot.py --epochs 10 --batch_size 64
```

## Quick start (Kaggle — free T4 GPU)

1. Create a new Kaggle notebook
2. Add dataset: `skin-lesion-analysis-toward-melanoma-detection`
3. Toggle GPU accelerator ON (T4 x1)
4. Follow the cells in `kaggle_run.py`
5. Expected runtime: ~25 minutes

## Key output metrics

| Metric | Meaning |
|---|---|
| `Δ EOD gap` | Change in equalized odds gap after CL. **Positive = fairness worsened.** |
| `Δ acc_male` | Accuracy drop for male patients after Task 2 fine-tuning |
| `Δ acc_female` | Accuracy drop for female patients after Task 2 fine-tuning |
| Differential forgetting | `|Δ acc_male - Δ acc_female|` — the core signal |

## Pilot go/no-go thresholds

| Signal | Threshold | Meaning |
|---|---|---|
| `Δ EOD gap > 0` | qualitative | Fairness degraded |
| Differential forgetting | `> 0.02` | Subgroups forgotten unequally |

## RadImageNet weights

Download from: https://github.com/BMEII-AI/RadImageNet

Place the ResNet50 `.pth` file at `./data/RadImageNet_resnet50.pth`
or set `RADIMAGENET_WEIGHTS` environment variable.

If weights are not found, the code falls back to ImageNet pretrained
weights with a warning (results will still be directionally valid
for the pilot).
