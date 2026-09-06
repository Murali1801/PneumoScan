"""Stage 1: fetch RSNA, apply CLAHE, build splits, scale the radiologist boxes.

    python scripts/prepare_data.py --download      # needs Kaggle creds + rules accepted
    python scripts/prepare_data.py                 # data already unzipped

IMPORTANT: RSNA is a Kaggle *competition*, not a dataset.  You must open
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
and click "I Understand and Accept" once, or the download returns 403.

Writes:
    data/processed/<split>/<CLS>/<patientId>.png   CLAHE'd, resized, lossless
    data/splits.csv                                manifest every later stage reads
    reports/dataset_summary.csv                    images and boxes per split
    reports/cohort.csv                             age / sex / view distribution
    reports/clahe_examples.png                     before/after figure, boxes drawn
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd

from config import Config
from data import (class_weights, detailed_summary, duplicate_images_across_splits,
                  find_dataset_root, index_rsna, make_splits, parse_boxes,
                  scale_boxes, summarise)
from localization import draw_boxes
from preprocess import (clahe_image, dicom_metadata, prepare_image,
                        preprocess_dataset, read_gray)

COMPETITION = "rsna-pneumonia-detection-challenge"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"


def download(raw_root: Path) -> Path:
    """Download + unzip the RSNA competition data (about 3.7 GB)."""
    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = raw_root / f"{COMPETITION}.zip"

    if not zip_path.exists():
        print(f"downloading {COMPETITION} (~3.7 GB) ...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(COMPETITION, path=str(raw_root), quiet=False)
        except Exception as exc:
            if "403" in str(exc) or "Forbidden" in str(exc):
                raise SystemExit(
                    f"\nKaggle returned 403 for {COMPETITION}.\n"
                    f"Open {RULES_URL}\n"
                    "and click 'I Understand and Accept', then rerun. This is a "
                    "one-off step Kaggle requires for every competition.\n") from exc
            print(f"Kaggle Python API failed ({exc}); trying the CLI ...")
            subprocess.run(["kaggle", "competitions", "download",
                            "-c", COMPETITION, "-p", str(raw_root)], check=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"expected {zip_path} after download")

    print("unzipping (about 30k DICOMs, takes a minute) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_root)
    return raw_root


def cohort_table(df: pd.DataFrame, n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Age / sex / view distribution, read straight from the DICOM headers.

    This is what lets the report *show* the cohort is adult rather than assert it.
    Sampled rather than exhaustive - header reads are cheap but not free.
    """
    sample = df.sample(min(n, len(df)), random_state=seed)
    meta = [dicom_metadata(p) for p in sample.path]
    m = pd.DataFrame(meta)
    ages = m.age.dropna()
    rows = [{
        "n_sampled": len(m),
        "age_min": int(ages.min()) if len(ages) else None,
        "age_median": float(ages.median()) if len(ages) else None,
        "age_max": int(ages.max()) if len(ages) else None,
        "pct_under_18": round(100.0 * (ages < 18).mean(), 2) if len(ages) else None,
        "pct_male": round(100.0 * (m.sex == "M").mean(), 2),
        "view_PA": round(100.0 * (m.view == "PA").mean(), 2),
        "view_AP": round(100.0 * (m.view == "AP").mean(), 2),
    }]
    return pd.DataFrame(rows)


def clahe_example_figure(df: pd.DataFrame, cfg: Config, out_path: Path, n: int = 3):
    """Original / CLAHE / boxes - a slide-ready figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = df[(df.split == "test") & (df.n_boxes > 0)].head(max(1, n - 1))
    neg = df[(df.split == "test") & (df.n_boxes == 0)].head(1)
    rows = list(pd.concat([pos, neg]).head(n).itertuples())
    if not rows:
        rows = list(df.head(n).itertuples())

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        raw = cv2.resize(read_gray(row.path), (cfg.img_size, cfg.img_size),
                         interpolation=cv2.INTER_AREA)
        eq = prepare_image(row.path, cfg.img_size, cfg.clahe_clip, cfg.clahe_grid)
        boxed = draw_boxes(cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB),
                           parse_boxes(row.boxes_scaled))
        panels = ((raw, "gray", "Original DICOM"),
                  (eq, "gray", f"CLAHE (clip={cfg.clahe_clip}, grid={cfg.clahe_grid})"),
                  (boxed, None, "Radiologist boxes"))
        for c, (img, cmap, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[r, 0].set_ylabel(f"{row.cls}\n({row.n_boxes} box)", fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("RSNA adult chest radiographs: CLAHE preprocessing", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="fetch the competition data from Kaggle first")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-cohort", action="store_true",
                    help="skip the DICOM header scan for the cohort table")
    Config.add_args(ap)
    args = ap.parse_args()
    cfg = Config.from_args(args)

    raw_root = Path(cfg.raw_dir)
    if args.download:
        download(raw_root)
    dataset_root = find_dataset_root(raw_root, raw_root.parent)
    print(f"dataset root: {dataset_root}")

    df = index_rsna(dataset_root, exclude_not_normal=cfg.exclude_not_normal)
    print(f"indexed {len(df):,} images, {int(df.n_boxes.sum()):,} radiologist boxes")

    # NB: --subsample-train is deliberately NOT applied here. splits.csv is the
    # canonical manifest; shrinking it would make every later run silently
    # smaller. Subsampling happens in train.py, at training time only.
    df = make_splits(df, cfg.val_frac, cfg.test_frac, cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("checking for byte-identical images across splits ...")
    dupes = duplicate_images_across_splits(df)
    if dupes.empty:
        print("  none found - no image appears in two splits")
    else:
        print(f"  WARNING: {dupes.md5.nunique()} image(s) in more than one split "
              f"({len(dupes)} files). Test scores may be optimistic.")
        dupes.to_csv(out / "duplicate_images.csv", index=False)

    # Boxes live in original DICOM pixel space; rescale into the resized frame.
    print("scaling boxes into the resized frame ...")
    sizes = {}
    for row in df.itertuples():
        if row.n_boxes:
            meta = dicom_metadata(row.path)
            sizes[row.path] = (meta["cols"] or 1024, meta["rows"] or 1024)
    df["boxes_scaled"] = [
        scale_boxes(r.boxes, *sizes.get(r.path, (1024, 1024)), cfg.img_size)
        if r.n_boxes else "[]" for r in df.itertuples()]

    print(f"\nCLAHE preprocessing -> {cfg.processed_dir}")
    rel = [f"{r.split}/{r.cls}/{r.patient}.png" for r in df.itertuples()]
    df["proc_path"] = preprocess_dataset(
        zip(df.path, rel), cfg.processed_dir,
        size=cfg.img_size, clip=cfg.clahe_clip, grid=cfg.clahe_grid,
        workers=args.workers)

    Path(cfg.splits_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.splits_csv, index=False)

    summary = summarise(df)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    detail = detailed_summary(df)
    detail.to_csv(out / "detailed_class_summary.csv")

    print("\n" + summary.to_string(index=False))
    print("\nRSNA class breakdown:")
    print(detail.to_string())
    print("\nclass weights (train):",
          {k: round(v, 3) for k, v in class_weights(df).items()})

    if not args.skip_cohort:
        print("\nreading DICOM headers for the cohort table ...")
        cohort = cohort_table(df, seed=cfg.seed)
        cohort.to_csv(out / "cohort.csv", index=False)
        print(cohort.to_string(index=False))

    fig = clahe_example_figure(df, cfg, out / "clahe_examples.png")
    print(f"\nwrote {cfg.splits_csv}")
    print(f"wrote {out / 'dataset_summary.csv'}, {out / 'detailed_class_summary.csv'}")
    print(f"wrote {fig}")


if __name__ == "__main__":
    main()
