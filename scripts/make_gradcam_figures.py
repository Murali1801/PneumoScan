"""Stage 3: Grad-CAM figures AND the quantitative localisation report.

    python scripts/make_gradcam_figures.py --backbone densenet121

Writes into reports/<run>/:
    gradcam_pneumonia.png    confident true positives, radiologist boxes drawn
    gradcam_normal.png       confident true negatives
    gradcam_errors.png       the model's mistakes - the most informative panel
    gradcam_vs_pp.png        Grad-CAM against Grad-CAM++ on the same images
    localization.json        pointing game / energy / IoU vs the boxes
    localization.csv         the same as a table, for pasting into the report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras

from config import Config
from data import parse_boxes
from gradcam import CAMExplainer, figure_grid, overlay
from localization import (compare_methods, draw_boxes, evaluate_localization,
                          format_localization)
from preprocess import to_model_input

BOX_COLOUR = (0, 255, 0)


def _load_batch(rows):
    """Read the already-CLAHE'd PNGs back as (greyscales, model-input batch)."""
    grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in rows.proc_path]
    batch = np.concatenate([to_model_input(g) for g in grays]).astype("float32")
    return grays, batch


def _case(row, gray, heat, prob) -> dict:
    """One figure row: CLAHE input | boxes drawn | Grad-CAM with boxes."""
    boxes = parse_boxes(row.boxes_scaled)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return {
        "original": gray,
        "clahe": draw_boxes(rgb, boxes, BOX_COLOUR),
        "overlay": draw_boxes(overlay(gray, heat), boxes, BOX_COLOUR),
        "prob": float(prob),
        "truth": row.cls,
        "name": f"{row.patient[:8]}...  ({row.n_boxes} box)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="cases per figure")
    ap.add_argument("--loc-max", type=int, default=1500,
                    help="cap images used for the localisation metrics (0 = all)")
    Config.add_args(ap)
    args = ap.parse_args()
    cfg = Config.from_args(args)

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"{cfg.model_path} not found - train the model first.")
    model = keras.models.load_model(cfg.model_path)
    cam = CAMExplainer(model, "gradcam")
    campp = CAMExplainer(model, "gradcam++")

    df = pd.read_csv(cfg.splits_csv)
    test = df[df.split == "test"].reset_index(drop=True)

    pred_path = cfg.run_dir / "test_predictions.npz"
    if pred_path.exists():
        blob = np.load(pred_path)
        probs, threshold = blob["y_prob"], float(blob["threshold"])
        if len(probs) != len(test):
            probs, threshold = None, 0.5
    else:
        probs, threshold = None, 0.5

    if probs is None:
        print("scoring the test set ...")
        scores = []
        for i in range(0, len(test), 32):
            _, batch = _load_batch(test.iloc[i:i + 32])
            scores.append(model.predict(batch, verbose=0).ravel())
        probs = np.concatenate(scores)

    test["prob"] = probs
    test["pred"] = np.where(test.prob >= threshold, "PNEUMONIA", "NORMAL")
    test["correct"] = test.pred == test.cls
    print(f"threshold {threshold:.3f} -> "
          f"{test.correct.sum():,}/{len(test):,} correct on the test set")

    # ---------------------------------------------------------------- figures
    picks = {
        "gradcam_pneumonia": test[(test.cls == "PNEUMONIA") & test.correct
                                  & (test.n_boxes > 0)].nlargest(args.n, "prob"),
        "gradcam_normal": test[(test.cls == "NORMAL") & test.correct]
                          .nsmallest(args.n, "prob"),
        "gradcam_errors": test[~test.correct]
                          .assign(margin=lambda d: (d.prob - threshold).abs())
                          .nlargest(args.n, "margin"),
    }
    titles = {
        "gradcam_pneumonia": "Grad-CAM vs radiologist boxes: true positives",
        "gradcam_normal": "Grad-CAM: confident NORMAL predictions",
        "gradcam_errors": "Grad-CAM: where the model got it wrong",
    }
    headers = ("CLAHE input", "Radiologist boxes", "Grad-CAM + boxes")

    index = {}
    for name, sub in picks.items():
        if sub.empty:
            print(f"skipping {name}: no such cases")
            continue
        grays, batch = _load_batch(sub)
        heats, p = cam.explain(batch, target="predicted")
        items = [_case(r, g, h, pr)
                 for r, g, h, pr in zip(sub.itertuples(), grays, heats, p)]
        out = figure_grid(items, cfg.run_dir / f"{name}.png",
                          threshold=threshold, title=titles[name], headers=headers)
        index[name] = [it["name"] for it in items]
        print(f"wrote {out}")

    sub = picks["gradcam_pneumonia"]
    if not sub.empty:
        grays, batch = _load_batch(sub)
        h_cam, p = cam.explain(batch, target="predicted")
        h_pp, _ = campp.explain(batch, target="predicted")
        items = []
        for r, g, a, b, pr in zip(sub.itertuples(), grays, h_cam, h_pp, p):
            boxes = parse_boxes(r.boxes_scaled)
            items.append({
                "original": draw_boxes(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), boxes, BOX_COLOUR),
                "clahe": draw_boxes(overlay(g, a), boxes, BOX_COLOUR),
                "overlay": draw_boxes(overlay(g, b), boxes, BOX_COLOUR),
                "prob": float(pr), "truth": r.cls,
                "name": f"{r.patient[:8]}...",
            })
        out = figure_grid(items, cfg.run_dir / "gradcam_vs_pp.png",
                          threshold=threshold,
                          title="Grad-CAM vs Grad-CAM++ (boxes in green)",
                          headers=("Boxes only", "Grad-CAM", "Grad-CAM++"))
        print(f"wrote {out}")

    # ------------------------------------------------------------- the metric
    boxed = test[(test.n_boxes > 0) & test.correct]
    if args.loc_max and len(boxed) > args.loc_max:
        boxed = boxed.sample(args.loc_max, random_state=cfg.seed)
    print(f"\nscoring localisation on {len(boxed):,} correctly-detected "
          "pneumonia cases ...")

    table = compare_methods({"Grad-CAM": cam, "Grad-CAM++": campp},
                            boxed, cfg, tau=cfg.iou_threshold)
    table.to_csv(cfg.run_dir / "localization.csv", index=False)
    report = {r["explainer"]: {k: v for k, v in r.items() if k != "explainer"}
              for r in table.to_dict("records")}
    (cfg.run_dir / "localization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print()
    for method, r in report.items():
        print(format_localization(r, f"{cfg.run_name} / {method}"))
        print()
    print(f"wrote {cfg.run_dir / 'localization.csv'}")

    (cfg.run_dir / "gradcam_index.json").write_text(
        json.dumps({"threshold": threshold, "figures": index}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
