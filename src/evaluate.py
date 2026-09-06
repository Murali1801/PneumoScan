"""Evaluation: threshold selection on val, metrics + curves on the held-out test set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def predict(model, ds):
    """Returns (y_true, y_prob) for a non-shuffled dataset."""
    probs = model.predict(ds, verbose=0).ravel()
    y = np.concatenate([b.numpy() for _, b in ds]).ravel()
    return y.astype(int), probs.astype(float)


def pick_threshold(y_true, y_prob, strategy: str = "youden") -> float:
    """Choose the operating point on the VALIDATION set only.

    0.5 is an arbitrary cut once the training set is imbalanced (RSNA is roughly
    3:1 against pneumonia).  Youden's J maximises sensitivity + specificity - 1,
    the right objective for a screening tool where a missed pneumonia and a
    false alarm both carry cost.  Tuning it on test would leak the test set.
    """
    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])
    if strategy == "f1":
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros_like(prec), where=(prec + rec) > 0)
        return float(thr[min(int(np.argmax(f1)), len(thr) - 1)])
    raise ValueError(f"unknown threshold strategy: {strategy!r}")


def bootstrap_auc(y_true, y_prob, n: int = 2000, seed: int = 0):
    """95% confidence interval for AUROC by bootstrap resampling of test images."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    scores = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        scores.append(roc_auc_score(y_true[s], y_prob[s]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def metrics_at(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_curves(y_true, y_prob, out_dir, threshold: float, prefix: str = "test"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(fpr, tpr, lw=2, color="#1b6ca8", label="AUROC = %.4f" % auroc)
    ax[0].plot([0, 1], [0, 1], "--", lw=1, color="#999999")
    ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC curve")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.25)

    ax[1].plot(rec, prec, lw=2, color="#c0392b", label="AUPRC = %.4f" % auprc)
    ax[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve")
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    paths["curves"] = out_dir / (prefix + "_curves.png")
    fig.savefig(paths["curves"], dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set(xticks=[0, 1], yticks=[0, 1],
              xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
              xlabel="predicted", ylabel="true",
              title="Confusion matrix (threshold = %.3f)" % threshold)
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].hist(y_prob[y_true == 0], bins=30, alpha=0.65, label="NORMAL", color="#1b6ca8")
    ax[1].hist(y_prob[y_true == 1], bins=30, alpha=0.65, label="PNEUMONIA", color="#c0392b")
    ax[1].axvline(threshold, color="black", ls="--", lw=1.5,
                  label="threshold %.3f" % threshold)
    ax[1].set(xlabel="predicted P(pneumonia)", ylabel="count", title="Score distribution")
    ax[1].legend()
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    paths["confusion"] = out_dir / (prefix + "_confusion.png")
    fig.savefig(paths["confusion"], dpi=150)
    plt.close(fig)
    return paths


def plot_history(history: dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in ("loss", "auc", "acc") if k in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        ax.plot(history[k], label="train " + k, lw=2)
        if "val_" + k in history:
            ax.plot(history["val_" + k], label="val " + k, lw=2)
        ax.set(xlabel="epoch", ylabel=k, title=k)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_evaluation(model, datasets, out_dir, threshold_strategy: str = "youden") -> dict:
    """val -> operating threshold, test -> full metric report.  Writes metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_val, p_val = predict(model, datasets["val"])
    thr = pick_threshold(y_val, p_val, threshold_strategy)

    y_test, p_test = predict(model, datasets["test"])
    report = {
        "val": metrics_at(y_val, p_val, thr),
        "test": metrics_at(y_test, p_test, thr),
        "test_at_0.5": metrics_at(y_test, p_test, 0.5),
        "test_auroc_95ci": bootstrap_auc(y_test, p_test),
        "n": {"val": int(len(y_val)), "test": int(len(y_test))},
    }
    plot_curves(y_test, p_test, out_dir, thr, prefix="test")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(out_dir / "test_predictions.npz",
             y_true=y_test, y_prob=p_test, threshold=thr)
    return report


def format_report(report: dict, name: str = "") -> str:
    t = report["test"]
    lo, hi = report["test_auroc_95ci"]
    cm = t["confusion_matrix"]
    return "\n".join([
        "=== %s - held-out test set (n=%d) ===" % (name, report["n"]["test"]),
        "  operating threshold : %.4f  (tuned on val, Youden's J)" % t["threshold"],
        "  AUROC               : %.4f  (95%% CI %.4f-%.4f)" % (t["auroc"], lo, hi),
        "  AUPRC               : %.4f" % t["auprc"],
        "  accuracy            : %.4f" % t["accuracy"],
        "  sensitivity (recall): %.4f" % t["sensitivity_recall"],
        "  specificity         : %.4f" % t["specificity"],
        "  precision (PPV)     : %.4f" % t["precision_ppv"],
        "  F1                  : %.4f" % t["f1"],
        "  confusion  tn=%d fp=%d fn=%d tp=%d" % (cm["tn"], cm["fp"], cm["fn"], cm["tp"]),
    ])
