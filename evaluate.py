# ─────────────────────────────────────────────────────────────
# evaluate.py  –  per-subgroup fairness metrics
#
# Core metrics:
#   acc    – accuracy
#   tpr    – true positive rate (sensitivity for melanoma)
#   fpr    – false positive rate
#   auc    – ROC AUC
#   eod_gap – equalized odds gap |TPR_male - TPR_female|
#   acc_gap – accuracy gap |acc_male - acc_female|
#
# All are computed separately per sex group and globally.
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config

SUBGROUP_MAP = {0: "female", 1: "male"}


def evaluate_fairness(
    model: nn.Module,
    loader,
    device: torch.device,
    split_name: str = "",
) -> dict:
    """
    Runs a full evaluation pass and returns a nested dict:

    {
      "overall": { acc, auc, tpr, fpr, n },
      "male":    { acc, auc, tpr, fpr, n },
      "female":  { acc, auc, tpr, fpr, n },
      "eod_gap": float,   # |TPR_male - TPR_female|
      "acc_gap": float,   # |acc_male - acc_female|
      "auc_gap": float,   # |auc_male - auc_female|
    }
    """
    model.eval()

    # Buffers: collect everything, split after
    all_labels    = []
    all_probs     = []
    all_subgroups = []

    with torch.no_grad():
        for imgs, labels, subgroups in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]  # P(melanoma)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subgroups.extend(subgroups.cpu().numpy())

    all_labels    = np.array(all_labels)
    all_probs     = np.array(all_probs)
    all_subgroups = np.array(all_subgroups)
    all_preds     = (all_probs >= config.DECISION_THRESHOLD).astype(int)

    # ── Compute per-group metrics ─────────────────────────────
    results = {}
    groups_to_eval = {"overall": None, "male": 1, "female": 0}

    for group_name, group_val in groups_to_eval.items():
        if group_val is None:
            mask = np.ones(len(all_labels), dtype=bool)
        else:
            mask = all_subgroups == group_val

        y_true = all_labels[mask]
        y_pred = all_preds[mask]
        y_prob = all_probs[mask]

        n = mask.sum()

        if n == 0:
            results[group_name] = {
                "acc": None, "tpr": None, "fpr": None,
                "auc": None, "n": 0
            }
            continue

        acc = (y_pred == y_true).mean()

        # TPR = sensitivity = P(pred=mel | true=mel)
        pos_mask = y_true == 1
        tpr = (y_pred[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0.0

        # FPR = P(pred=mel | true=nv)
        neg_mask = y_true == 0
        fpr = (y_pred[neg_mask] == 1).mean() if neg_mask.sum() > 0 else 0.0

        # AUC (only meaningful if both classes present)
        auc = None
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)

        results[group_name] = {
            "acc": float(acc),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "auc": float(auc) if auc is not None else None,
            "n":   int(n),
        }

    # ── Fairness gaps ─────────────────────────────────────────
    # Equalized Odds Gap: difference in TPR between groups
    # A positive Δ after CL means one group is being "forgotten" faster
    male_tpr   = results["male"]["tpr"]   or 0.0
    female_tpr = results["female"]["tpr"] or 0.0
    male_acc   = results["male"]["acc"]   or 0.0
    female_acc = results["female"]["acc"] or 0.0
    male_auc   = results["male"]["auc"]   or 0.0
    female_auc = results["female"]["auc"] or 0.0

    results["eod_gap"] = abs(male_tpr   - female_tpr)
    results["acc_gap"] = abs(male_acc   - female_acc)
    results["auc_gap"] = abs(male_auc   - female_auc)

    _print_metrics(results, split_name)
    return results


def _print_metrics(results: dict, split_name: str):
    label = f"[eval{' ' + split_name if split_name else ''}]"

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    print(f"\n{label} {'Group':<8} {'n':>5}  {'acc':>6}  "
          f"{'tpr':>6}  {'fpr':>6}  {'auc':>6}")
    print(f"{label} {'-'*50}")
    for g in ["overall", "male", "female"]:
        m = results[g]
        print(f"{label} {g:<8} {m['n']:>5}  {fmt(m['acc'])}  "
              f"{fmt(m['tpr'])}  {fmt(m['fpr'])}  {fmt(m['auc'])}")

    print(f"{label} {'─'*50}")
    print(f"{label} EOD gap  (|TPR_m - TPR_f|): {results['eod_gap']:.4f}")
    print(f"{label} Acc gap  (|acc_m - acc_f|): {results['acc_gap']:.4f}")
    print(f"{label} AUC gap  (|auc_m - auc_f|): {results['auc_gap']:.4f}")


def delta_metrics(before: dict, after: dict) -> dict:
    """
    Computes the change in all key metrics between two evaluation dicts.
    Positive Δ means the metric increased (usually bad for forgetting,
    good for gap reduction depending on the metric).
    """
    deltas = {}
    for group in ["overall", "male", "female"]:
        deltas[group] = {}
        for key in ["acc", "tpr", "fpr", "auc"]:
            b = before[group][key]
            a = after[group][key]
            deltas[group][key] = (a - b) if (a is not None and b is not None) else None

    deltas["eod_gap"] = after["eod_gap"] - before["eod_gap"]
    deltas["acc_gap"] = after["acc_gap"] - before["acc_gap"]
    deltas["auc_gap"] = after["auc_gap"] - before["auc_gap"]
    return deltas


def print_delta_report(before: dict, after: dict, task_label: str = "Task 2"):
    """
    Prints a clean forgetting report comparing before/after a CL step.
    This is the core output that tells you if the pilot hypothesis holds.
    """
    d = delta_metrics(before, after)

    print(f"\n{'═'*55}")
    print(f"  FORGETTING REPORT after fine-tuning on {task_label}")
    print(f"{'═'*55}")
    print(f"  {'Metric':<28} {'Before':>7}  {'After':>7}  {'Δ':>7}")
    print(f"  {'─'*48}")

    rows = [
        ("Overall acc",      "overall", "acc"),
        ("Overall TPR",      "overall", "tpr"),
        ("Overall AUC",      "overall", "auc"),
        ("Male   acc",       "male",    "acc"),
        ("Male   TPR",       "male",    "tpr"),
        ("Female acc",       "female",  "acc"),
        ("Female TPR",       "female",  "tpr"),
    ]

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    def fmt_d(v):
        if v is None:
            return "  N/A"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.4f}"

    for label, group, key in rows:
        b_val = before[group][key]
        a_val = after[group][key]
        d_val = d[group][key]
        print(f"  {label:<28} {fmt(b_val)}  {fmt(a_val)}  {fmt_d(d_val)}")

    print(f"  {'─'*48}")
    print(f"  {'EOD gap |TPR_m - TPR_f|':<28} "
          f"{before['eod_gap']:.4f}  {after['eod_gap']:.4f}  "
          f"{fmt_d(d['eod_gap'])}")
    print(f"  {'Acc gap |acc_m - acc_f|':<28} "
          f"{before['acc_gap']:.4f}  {after['acc_gap']:.4f}  "
          f"{fmt_d(d['acc_gap'])}")
    print(f"{'═'*55}")

    # ── Go / No-go signal ─────────────────────────────────────
    diff_forgetting = abs(
        (d["male"]["acc"]   or 0.0) -
        (d["female"]["acc"] or 0.0)
    )
    eod_worsened = d["eod_gap"] > 0

    print(f"\n  PILOT VERDICT")
    print(f"  Differential forgetting (acc):  {diff_forgetting:.4f}  "
          f"{'✅' if diff_forgetting > 0.02 else '⚠️ weak signal'}")
    print(f"  EOD gap worsened after CL:      "
          f"{'yes' if eod_worsened else 'no'}  "
          f"{'✅' if eod_worsened else '⚠️ no degradation'}")

    if diff_forgetting > 0.02 and eod_worsened:
        print(f"\n  🚀 HYPOTHESIS CONFIRMED – proceed to full experiment")
    elif diff_forgetting > 0.02 or eod_worsened:
        print(f"\n  🔶 PARTIAL SIGNAL – worth investigating further")
    else:
        print(f"\n  🔴 WEAK SIGNAL – consider different task split or dataset")
    print(f"{'═'*55}\n")
