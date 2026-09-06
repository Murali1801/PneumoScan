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


def numbering_pool(src_split: str) -> str:
    """Which patient-numbering namespace a source folder belongs to.

    Kermany numbers patients independently inside the official `test` folder:
    `train/PNEUMONIA/person1_bacteria_1.jpeg` and
    `test/PNEUMONIA/person1_virus_6.jpeg` are two different children who happen
    to share an index.  `train` and `val` were drawn from one pool and do share
    numbering, so they stay in one namespace - that way a patient appearing in
    both is correctly recognised as the same person when the two are merged.
    """
    return "test" if src_split == "test" else "trainval"


def patient_id(filename: str, cls: str, src_split: str = "trainval") -> str:
    """Best-effort patient/study id, namespaced by numbering pool and class.

    Falls back to the filename stem, which makes the image its own group - that
    is the safe direction: it never merges two different patients.
    """
    stem = Path(filename).stem
    gid = stem.upper()
    for pat in _PATIENT_PATTERNS:
        m = pat.match(stem)
        if m:
            gid = m.group(1).upper()
            break
    return f"{numbering_pool(src_split)}:{cls}:{gid}"


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
                    "patient": patient_id(f.name, cls, src_split),
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
    """Guard the train/val split - the only split this code actually makes.

    train vs test cannot collide here by construction, because `numbering_pool`
    puts them in different namespaces.  Their separation rests on Kermany having
    sampled the test folder from different patients, which is a property of the
    dataset, not something filenames can prove.  `duplicate_images_across_splits`
    is the check that actually interrogates it.
    """
    groups = {s: set(g.patient) for s, g in df.groupby("split")}
    shared = groups.get("train", set()) & groups.get("val", set())
    if shared:
        raise AssertionError(
            f"patient leakage between train and val: {sorted(shared)[:5]} ... "
            f"({len(shared)} groups) - StratifiedGroupKFold should make this "
            "impossible; check that `patient` is being passed as `groups`.")


def duplicate_images_across_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical images appearing in more than one split.

    Patient ids come from filenames, so they cannot detect the same radiograph
    filed twice under different names.  Hashing the actual file contents can.
    Cheap (a few seconds for 5,856 files) and worth reporting: a duplicate
    spanning train and test would inflate the headline test score.
    """
    import hashlib

    digests = []
    for path in df.path:
        with open(path, "rb") as fh:
            digests.append(hashlib.md5(fh.read()).hexdigest())
    tmp = df.assign(md5=digests)

    spread = tmp.groupby("md5").split.nunique()
    offenders = spread[spread > 1].index
    return (tmp[tmp.md5.isin(offenders)]
            .sort_values(["md5", "split"])[["md5", "split", "cls", "path"]])


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
