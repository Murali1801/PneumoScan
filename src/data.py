"""Indexing and patient-grouped splitting of the Kermany chest X-ray dataset.

Pure pandas/sklearn - no TensorFlow - so the split can be inspected and unit
tested anywhere.

Why patient grouping matters
----------------------------
Kermany ships several images per patient (`person23_bacteria_76.jpeg` and
`person23_bacteria_77.jpeg` are the same child).  A naive random validation
split puts one of a patient's images in train and another in val, so the model
can score well by memorising that patient rather than by learning pneumonia.
Every split here is made over *patient groups*, never over individual files.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
IMG_EXT = {".jpeg", ".jpg", ".png", ".bmp"}

# Kermany filename conventions, most specific first.
_PATIENT_PATTERNS = (
    re.compile(r"^(NORMAL\d*-IM-\d+)", re.I),          # NORMAL2-IM-1427-0001.jpeg
    re.compile(r"^(IM-\d+)", re.I),                    # IM-0115-0001.jpeg
    re.compile(r"^(person\d+)", re.I),                 # person1_bacteria_1.jpeg
    re.compile(r"^((?:BACTERIA|VIRUS)-\d+)", re.I),    # BACTERIA-1135262-0001.jpeg
)


def patient_id(filename: str, cls: str) -> str:
    """Best-effort patient/study id, namespaced by class.

    Falls back to the filename stem, which makes the image its own group - that
    is the safe direction: it never merges two different patients.
    """
    stem = Path(filename).stem
    for pat in _PATIENT_PATTERNS:
        m = pat.match(stem)
        if m:
            return f"{cls}:{m.group(1).upper()}"
    return f"{cls}:{stem.upper()}"


def index_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Walk `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` into a DataFrame."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found - download the Kermany dataset first "
            "(see scripts/prepare_data.py --help)")

    rows = []
    for src_split in ("train", "val", "test"):
        for cls in CLASS_TO_LABEL:
            folder = raw_dir / src_split / cls
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in IMG_EXT or f.name.startswith("."):
                    continue
                rows.append({
                    "path": str(f),
                    "filename": f.name,
                    "cls": cls,
                    "label": CLASS_TO_LABEL[cls],
                    "src_split": src_split,
                    "patient": patient_id(f.name, cls),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no images found under {raw_dir}")
    return df


def make_splits(df: pd.DataFrame, val_frac: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Add a `split` column: train / val / test.

    - The official `test` folder is held out untouched: it is the only honest
      estimate of generalisation, and its patients never appear in training.
    - The official `val` folder holds just 16 images (8 per class), far too few
      to select a model or a decision threshold on, so it is merged back into
      train and a proper ~10% validation set is carved out with
      StratifiedGroupKFold - class-balanced *and* patient-disjoint.
    """
    df = df.copy()
    df["split"] = "test"

    pool = df[df.src_split.isin(("train", "val"))]
    n_splits = max(2, round(1 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(pool, y=pool.label, groups=pool.patient))

    df.loc[pool.index[tr_idx], "split"] = "train"
    df.loc[pool.index[va_idx], "split"] = "val"

    _assert_no_leakage(df)
    return df


def _assert_no_leakage(df: pd.DataFrame) -> None:
    groups = {s: set(g.patient) for s, g in df.groupby("split")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = groups.get(a, set()) & groups.get(b, set())
        if shared:
            raise AssertionError(
                f"patient leakage between {a} and {b}: {sorted(shared)[:5]} ...")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Images and patients per split/class - the table for the report."""
    g = (df.groupby(["split", "cls"])
           .agg(images=("path", "size"), patients=("patient", "nunique"))
           .reset_index())
    order = {"train": 0, "val": 1, "test": 2}
    return g.sort_values(["split", "cls"], key=lambda s: s.map(order).fillna(s))


def class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights computed on the training split only."""
    tr = df[df.split == "train"]
    n = len(tr)
    counts = tr.label.value_counts().to_dict()
    return {int(k): n / (2.0 * v) for k, v in counts.items()}
