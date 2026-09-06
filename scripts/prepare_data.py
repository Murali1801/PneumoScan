"""Stage 1 of the pipeline: fetch Kermany, apply CLAHE, build patient-safe splits.

    python scripts/prepare_data.py --download          # needs a Kaggle API token
    python scripts/prepare_data.py                     # data already unzipped

Writes:
    data/processed/<split_src>/<CLASS>/<file>.png   CLAHE'd, resized, lossless
    data/splits.csv                                 the manifest every stage reads
    reports/dataset_summary.csv                     images + patients per split
    reports/clahe_examples.png                      before/after figure for the report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd

from config import Config
from data import (class_weights, duplicate_images_across_splits, index_raw,
                 make_splits, summarise)
from preprocess import prepare_image, preprocess_dataset, read_gray, clahe_image

KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"


def download(raw_root: Path) -> Path:
    """Download + unzip the Kermany dataset (about 1.2 GB).

    Needs Kaggle credentials: either ~/.kaggle/kaggle.json (Kaggle -> Settings
    -> API -> Create New Token) or the KAGGLE_USERNAME / KAGGLE_KEY env vars.
    Uses the Python API and falls back to the `kaggle` console script.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = raw_root / "chest-xray-pneumonia.zip"

    if not zip_path.exists():
        print(f"downloading {KAGGLE_SLUG} (~1.2 GB) ...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(KAGGLE_SLUG, path=str(raw_root),
                                       unzip=False, quiet=False)
        except Exception as exc:
            print(f"Kaggle Python API failed ({exc}); trying the CLI ...")
            subprocess.run(["kaggle", "datasets", "download",
                            "-d", KAGGLE_SLUG, "-p", str(raw_root)], check=True)
    if not zip_path.exists():
        raise FileNotFoundError(f"expected {zip_path} after download")

    print("unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_root)
    return raw_root


def find_dataset_root(*candidates: Path) -> Path:
    """The Kaggle archive nests `chest_xray/chest_xray/...` in some versions."""
    seen = []
    for root in candidates:
        if not root.exists():
            continue
        for cand in (root, root / "chest_xray", root / "chest_xray" / "chest_xray"):
            seen.append(cand)
            if (cand / "train" / "NORMAL").is_dir():
                return cand
        for p in root.rglob("train/NORMAL"):
            if p.is_dir():
                return p.parent.parent
    raise FileNotFoundError(
        "could not find chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}; looked in:\n  "
        + "\n  ".join(str(s) for s in seen)
        + "\nRun with --download, or point --raw-dir at the unzipped folder.")


def clahe_example_figure(df: pd.DataFrame, cfg: Config, out_path: Path, n: int = 3):
    """Side-by-side original / CLAHE / difference - a slide-ready figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = df[df.split == "test"]
    per_class = max(1, n // 2)
    picks = pd.concat([test[test.cls == c].head(per_class) for c in ("NORMAL", "PNEUMONIA")])
    rows = list((picks if not picks.empty else df.head(n)).head(n).itertuples())

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        raw = cv2.resize(read_gray(row.path), (cfg.img_size, cfg.img_size),
                         interpolation=cv2.INTER_AREA)
        eq = prepare_image(row.path, cfg.img_size, cfg.clahe_clip, cfg.clahe_grid)
        diff = cv2.absdiff(eq, raw)
        for c, (img, cmap, title) in enumerate((
                (raw, "gray", "Original"),
                (eq, "gray", f"CLAHE (clip={cfg.clahe_clip}, grid={cfg.clahe_grid})"),
                (diff, "magma", "Contrast added"))):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[r, 0].set_ylabel(row.cls, fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=30)
    fig.suptitle("CLAHE preprocessing of chest radiographs", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="fetch the dataset from Kaggle first")
    ap.add_argument("--workers", type=int, default=8)
    Config.add_args(ap)
    args = ap.parse_args()
    cfg = Config.from_args(args)

    raw_root = Path(cfg.raw_dir)
    if args.download:
        download(raw_root.parent)
    dataset_root = find_dataset_root(raw_root, raw_root.parent)
    print(f"dataset root: {dataset_root}")

    df = index_raw(dataset_root)
    print(f"indexed {len(df):,} images, {df.patient.nunique():,} patient groups")

    df = make_splits(df, cfg.val_frac, cfg.seed)

    print("checking for byte-identical images across splits ...")
    dupes = duplicate_images_across_splits(df)
    if dupes.empty:
        print("  none found - no image appears in two splits")
    else:
        n = dupes.md5.nunique()
        print(f"  WARNING: {n} image(s) appear in more than one split "
              f"({len(dupes)} files). Test scores may be optimistic.")
        print(dupes.head(10).to_string(index=False))
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        dupes.to_csv(Path(cfg.out_dir) / "duplicate_images.csv", index=False)

    print(f"\nCLAHE preprocessing -> {cfg.processed_dir}")
    rel = [f"{r.src_split}/{r.cls}/{Path(r.filename).stem}.png" for r in df.itertuples()]
    df["proc_path"] = preprocess_dataset(
        zip(df.path, rel), cfg.processed_dir,
        size=cfg.img_size, clip=cfg.clahe_clip, grid=cfg.clahe_grid,
        workers=args.workers)

    Path(cfg.splits_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.splits_csv, index=False)

    summary = summarise(df)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "dataset_summary.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print("\nclass weights (train):",
          {k: round(v, 3) for k, v in class_weights(df).items()})
    print("no patient appears in more than one split - checked in make_splits()")

    fig = clahe_example_figure(df, cfg, out / "clahe_examples.png")
    print(f"\nwrote {cfg.splits_csv}")
    print(f"wrote {out / 'dataset_summary.csv'}")
    print(f"wrote {fig}")


if __name__ == "__main__":
    main()
