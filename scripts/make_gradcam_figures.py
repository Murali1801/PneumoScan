"""Stage 3: Grad-CAM explanation figures for the trained model.

    python scripts/make_gradcam_figures.py --backbone densenet121

Writes into reports/<run>/:
    gradcam_pneumonia.png   confident true-positive cases
    gradcam_normal.png      confident true-negative cases
    gradcam_errors.png      the model's mistakes - the most informative panel
    gradcam_vs_pp.png       Grad-CAM against Grad-CAM++ on the same images
    gradcam_index.json      which files ended up in which figure
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
from gradcam import CAMExplainer, figure_grid, overlay
from preprocess import prepare_image, read_gray


def _case(row, gray_clahe, heat, prob) -> dict:
    original = cv2.resize(read_gray(row.path), gray_clahe.shape[::-1],
                          interpolation=cv2.INTER_AREA)
    return {
        "original": original,
        "clahe": gray_clahe,
        "overlay": overlay(gray_clahe, heat),
        "prob": float(prob),
        "truth": row.cls,
        "name": Path(row.path).name,
    }


def _load_batch(rows, cfg: Config):
    grays = [prepare_image(r.path, cfg.img_size, cfg.clahe_clip, cfg.clahe_grid)
             for r in rows]
    batch = np.stack([cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) for g in grays]).astype("float32")
    return grays, batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="cases per figure")
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

    # Score the whole test set once so cases can be picked by confidence.
    metrics_path = cfg.run_dir / "test_predictions.npz"
    if metrics_path.exists():
        probs = np.load(metrics_path)["y_prob"]
        threshold = float(np.load(metrics_path)["threshold"])
        if len(probs) != len(test):
            probs, threshold = None, 0.5
    else:
        probs, threshold = None, 0.5

    if probs is None:
        print("scoring the test set ...")
        scores = []
        for i in range(0, len(test), 32):
            rows = list(test.iloc[i:i + 32].itertuples())
            _, batch = _load_batch(rows, cfg)
            scores.append(model.predict(batch, verbose=0).ravel())
        probs = np.concatenate(scores)

    test["prob"] = probs
    test["pred"] = np.where(test.prob >= threshold, "PNEUMONIA", "NORMAL")
    test["correct"] = test.pred == test.cls
    print(f"threshold {threshold:.3f} -> "
          f"{test.correct.sum()}/{len(test)} correct on the test set")

    picks = {
        "gradcam_pneumonia": test[(test.cls == "PNEUMONIA") & test.correct]
                             .nlargest(args.n, "prob"),
        "gradcam_normal": test[(test.cls == "NORMAL") & test.correct]
                          .nsmallest(args.n, "prob"),
        "gradcam_errors": test[~test.correct]
                          .assign(margin=lambda d: (d.prob - threshold).abs())
                          .nlargest(args.n, "margin"),
    }

    index = {}
    for name, sub in picks.items():
        if sub.empty:
            print(f"skipping {name}: no such cases (nice problem to have)")
            continue
        rows = list(sub.itertuples())
        grays, batch = _load_batch(rows, cfg)
        heats, p = cam.explain(batch, target="predicted")
        items = [_case(r, g, h, pr) for r, g, h, pr in zip(rows, grays, heats, p)]
        titles = {
            "gradcam_pneumonia": "Grad-CAM: confident PNEUMONIA predictions",
            "gradcam_normal": "Grad-CAM: confident NORMAL predictions",
            "gradcam_errors": "Grad-CAM: where the model got it wrong",
        }
        out = figure_grid(items, cfg.run_dir / f"{name}.png",
                          threshold=threshold, title=titles[name])
        index[name] = [it["name"] for it in items]
        print(f"wrote {out}")

    # Grad-CAM vs Grad-CAM++ on the same true-positive cases.
    sub = picks["gradcam_pneumonia"]
    if not sub.empty:
        rows = list(sub.itertuples())
        grays, batch = _load_batch(rows, cfg)
        h_cam, p = cam.explain(batch, target="predicted")
        h_pp, _ = campp.explain(batch, target="predicted")
        items = []
        for r, g, a, b, pr in zip(rows, grays, h_cam, h_pp, p):
            items.append({"original": g, "clahe": overlay(g, a),
                          "overlay": overlay(g, b), "prob": float(pr),
                          "truth": r.cls, "name": Path(r.path).name})
        out = figure_grid(items, cfg.run_dir / "gradcam_vs_pp.png",
                          threshold=threshold,
                          title="Grad-CAM vs Grad-CAM++ (same cases)",
                          headers=("CLAHE input", "Grad-CAM", "Grad-CAM++"))
        print(f"wrote {out}")
        index["gradcam_vs_pp"] = [it["name"] for it in items]

    (cfg.run_dir / "gradcam_index.json").write_text(
        json.dumps({"threshold": threshold, "figures": index}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
