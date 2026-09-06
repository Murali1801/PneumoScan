"""Does Grad-CAM look where the radiologist looked?

RSNA ships bounding boxes drawn by radiologists around each lung opacity, which
turns explainability from "here is a heatmap, it looks plausible" into a number.
Three complementary metrics:

**Pointing game** - does the single hottest pixel of the heatmap fall inside a
ground-truth box?  Simple and standard, but it only looks at one pixel.

**Energy-based pointing game** - what fraction of the heatmap's total mass lies
inside the boxes?  Introduced with Score-CAM; far less brittle than the argmax,
because a map that is broadly correct but peaks a few pixels outside the box is
not punished as a total miss.

**IoU at a threshold** - binarise the heatmap at tau x max, then intersect with
the boxes.  Rewards a map that is tight as well as correctly placed.

The number that makes all of this meaningful is the **chance baseline**: boxes
cover roughly a quarter of a chest film, so a heatmap pointing at random already
scores about 25% on the pointing game.  Every result here is reported against
that baseline - a metric without it is not evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def boxes_to_mask(boxes: np.ndarray, size: int) -> np.ndarray:
    """[x, y, w, h] boxes -> boolean (size, size) mask of their union."""
    mask = np.zeros((size, size), dtype=bool)
    for x, y, w, h in np.asarray(boxes, dtype="float32").reshape(-1, 4):
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(size, x1), min(size, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def upsample(heat: np.ndarray, size: int) -> np.ndarray:
    """Feature-map-resolution heatmap -> (size, size), values kept in [0, 1]."""
    out = cv2.resize(heat.astype("float32"), (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------
# metrics (one image)
# --------------------------------------------------------------------------
def pointing_hit(heat_full: np.ndarray, mask: np.ndarray) -> bool:
    """Is the hottest pixel inside the ground-truth region?"""
    y, x = np.unravel_index(int(np.argmax(heat_full)), heat_full.shape)
    return bool(mask[y, x])


def energy_fraction(heat_full: np.ndarray, mask: np.ndarray) -> float:
    """Share of total heatmap mass falling inside the ground-truth region."""
    total = float(heat_full.sum())
    if total <= 0:
        return 0.0
    return float(heat_full[mask].sum() / total)


def iou_at(heat_full: np.ndarray, mask: np.ndarray, tau: float = 0.5) -> float:
    """IoU between the boxes and the heatmap thresholded at tau x its maximum."""
    peak = float(heat_full.max())
    if peak <= 0:
        return 0.0
    pred = heat_full >= tau * peak
    union = np.logical_or(pred, mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, mask).sum() / union)


# --------------------------------------------------------------------------
# evaluation over a set of images
# --------------------------------------------------------------------------
def evaluate_localization(explainer, rows: pd.DataFrame, cfg, *,
                          batch_size: int = 32, tau: float = 0.5,
                          target: str = "pneumonia") -> dict:
    """Run the explainer over `rows` and score its maps against their boxes.

    `rows` must carry `proc_path` and `boxes_scaled` (boxes already in the
    resized coordinate frame).  Only images that actually have boxes are scored -
    a negative film has nothing to localise.
    """
    from preprocess import prepare_image, to_model_input
    from data import parse_boxes

    rows = rows[rows.n_boxes > 0]
    if rows.empty:
        return {"n": 0, "note": "no boxed images in this selection"}

    size = cfg.img_size
    hits, energies, ious, coverages, probs = [], [], [], [], []

    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in chunk.proc_path]
        batch = np.concatenate([to_model_input(g) for g in grays]).astype("float32")
        heats, p = explainer.explain(batch, target=target)

        for heat, boxes_json, prob in zip(heats, chunk.boxes_scaled, p):
            boxes = parse_boxes(boxes_json)
            mask = boxes_to_mask(boxes, size)
            if not mask.any():
                continue
            full = upsample(heat, size)
            hits.append(pointing_hit(full, mask))
            energies.append(energy_fraction(full, mask))
            ious.append(iou_at(full, mask, tau))
            coverages.append(float(mask.mean()))   # chance baseline for this image
            probs.append(float(prob))

    n = len(hits)
    if n == 0:
        return {"n": 0, "note": "no valid boxes after scaling"}

    chance = float(np.mean(coverages))
    return {
        "n": n,
        "method": explainer.mode,
        "pointing_game": float(np.mean(hits)),
        "pointing_game_chance": chance,
        "pointing_game_lift": float(np.mean(hits) / chance) if chance else None,
        "energy_pointing_game": float(np.mean(energies)),
        "energy_pointing_game_chance": chance,
        "iou@%.2f" % tau: float(np.mean(ious)),
        "mean_box_coverage": chance,
        "mean_confidence": float(np.mean(probs)),
    }


def compare_methods(explainers: dict, rows: pd.DataFrame, cfg, **kw) -> pd.DataFrame:
    """Score several explainers on the same images -> a table for the report."""
    out = []
    for name, ex in explainers.items():
        r = evaluate_localization(ex, rows, cfg, **kw)
        r["explainer"] = name
        out.append(r)
    df = pd.DataFrame(out)
    front = ["explainer", "n", "pointing_game", "pointing_game_chance",
             "pointing_game_lift", "energy_pointing_game"]
    cols = [c for c in front if c in df.columns] + \
           [c for c in df.columns if c not in front]
    return df[cols]


def format_localization(report: dict, name: str = "") -> str:
    if not report.get("n"):
        return f"=== {name} localisation === {report.get('note', 'no data')}"
    tau_key = next((k for k in report if k.startswith("iou@")), None)
    return "\n".join([
        f"=== {name} - localisation vs radiologist boxes (n={report['n']}) ===",
        "  pointing game        : %.3f   (chance %.3f, lift x%.2f)" % (
            report["pointing_game"], report["pointing_game_chance"],
            report["pointing_game_lift"] or 0.0),
        "  energy pointing game : %.3f   (chance %.3f)" % (
            report["energy_pointing_game"], report["energy_pointing_game_chance"]),
        "  %-21s: %.3f" % (tau_key, report[tau_key]) if tau_key else "",
        "  mean box coverage    : %.3f of the image" % report["mean_box_coverage"],
    ]).replace("\n\n", "\n")


def draw_boxes(rgb: np.ndarray, boxes, colour=(0, 255, 0), thickness: int = 2):
    """Draw ground-truth boxes onto an RGB uint8 image (returns a copy)."""
    out = rgb.copy()
    for x, y, w, h in np.asarray(boxes, dtype="float32").reshape(-1, 4):
        cv2.rectangle(out, (int(round(x)), int(round(y))),
                      (int(round(x + w)), int(round(y + h))), colour, thickness)
    return out
