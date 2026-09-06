"""Indexing and splitting of the RSNA Pneumonia Detection Challenge dataset.

Pure pandas/sklearn - no TensorFlow - so the split can be inspected anywhere.

What RSNA gives us
------------------
    stage_2_train_images/<patientId>.dcm        26,684 adult chest radiographs
    stage_2_train_labels.csv                    patientId, x, y, width, height, Target
    stage_2_detailed_class_info.csv             patientId, class

`stage_2_train_labels.csv` holds **one row per bounding box**, so a patient with
two opacities appears twice; a negative patient appears once with empty box
columns.  This module collapses that to one row per image, carrying the boxes
along as a list - they are what makes the Grad-CAM localisation metric possible.

The competition's own `stage_2_test_images/` is unlabelled (it was the leaderboard
set), so it is useless to us.  We split the 26,684 labelled images ourselves.

Label definition
----------------
`Target` is already binary: 1 = "Lung Opacity" (radiographic pneumonia),
0 = everything else.  Note that the 0s contain two very different populations -
truly normal films, and films that are abnormal for some *other* reason.  Keeping
the latter as negatives is the clinically honest choice: a screening tool has to
tell pneumonia apart from other pathology, not just from healthy lungs.  Set
`--exclude-not-normal` to drop them and see how much of your score was coming
from that easier task.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

TRAIN_IMAGES = "stage_2_train_images"
LABELS_CSV = "stage_2_train_labels.csv"
CLASS_CSV = "stage_2_detailed_class_info.csv"

NOT_NORMAL = "No Lung Opacity / Not Normal"


def find_dataset_root(*candidates: Path) -> Path:
    """Locate the folder holding stage_2_train_images/ and the two CSVs."""
    seen = []
    for root in candidates:
        root = Path(root)
        if not root.exists():
            continue
        for cand in (root, root / "rsna-pneumonia-detection-challenge"):
            seen.append(cand)
            if (cand / TRAIN_IMAGES).is_dir() and (cand / LABELS_CSV).is_file():
                return cand
        for p in root.rglob(TRAIN_IMAGES):
            if p.is_dir() and (p.parent / LABELS_CSV).is_file():
                return p.parent
    raise FileNotFoundError(
        f"could not find {TRAIN_IMAGES}/ next to {LABELS_CSV}; looked in:\n  "
        + "\n  ".join(str(s) for s in seen)
        + "\nRun with --download, or point --raw-dir at the unzipped folder.")


def index_rsna(raw_dir: str | Path, exclude_not_normal: bool = False) -> pd.DataFrame:
    """Collapse the per-box CSVs into one row per image, boxes carried along."""
    root = Path(raw_dir)
    labels = pd.read_csv(root / LABELS_CSV)
    detail = pd.read_csv(root / CLASS_CSV).drop_duplicates("patientId")

    boxes: dict[str, list] = {}
    for row in labels.itertuples():
        if row.Target == 1 and not any(
                pd.isna(v) for v in (row.x, row.y, row.width, row.height)):
            boxes.setdefault(row.patientId, []).append(
                [float(row.x), float(row.y), float(row.width), float(row.height)])

    per_image = (labels.groupby("patientId", as_index=False)
                       .Target.max()
                       .merge(detail, on="patientId", how="left"))

    img_dir = root / TRAIN_IMAGES
    per_image["path"] = per_image.patientId.map(lambda p: str(img_dir / f"{p}.dcm"))
    per_image["boxes"] = per_image.patientId.map(lambda p: json.dumps(boxes.get(p, [])))
    per_image["n_boxes"] = per_image.patientId.map(lambda p: len(boxes.get(p, [])))
    per_image["label"] = per_image.Target.astype(int)
    per_image["cls"] = per_image.label.map(LABEL_TO_CLASS)
    per_image = per_image.rename(columns={"patientId": "patient", "class": "detailed_class"})

    missing = [p for p in per_image.path.head(50) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{img_dir} does not contain the expected DICOMs, e.g. {missing[0]}")

    if exclude_not_normal:
        before = len(per_image)
        per_image = per_image[per_image.detailed_class != NOT_NORMAL].copy()
        print(f"  dropped {before - len(per_image):,} '{NOT_NORMAL}' images")

    keep = ["patient", "path", "label", "cls", "detailed_class", "boxes", "n_boxes"]
    return per_image[keep].reset_index(drop=True)


def make_splits(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                seed: int = 42) -> pd.DataFrame:
    """Add a `split` column: train / val / test, stratified on the label.

    RSNA issues one `patientId` per image, so grouping is one-image-per-group
    here and the group machinery is really a safety net.  It is kept because the
    guarantee we want - no patient on both sides of a split - should be enforced
    by the code rather than assumed from a dataset property.

    Honest caveat for the report: RSNA is re-annotated from NIH ChestX-ray, where
    one person can contribute several studies.  RSNA does not publish that
    mapping, so a small amount of same-person leakage is not detectable from this
    data alone.
    """
    df = df.copy()
    holdout = val_frac + test_frac
    if not 0 < holdout < 1:
        raise ValueError("val_frac + test_frac must be between 0 and 1")

    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / holdout)),
                                shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, y=df.label, groups=df.patient))

    df["split"] = "train"
    hold = df.iloc[hold_idx]

    # Divide the held-out block into val and test in the requested proportion.
    inner = StratifiedGroupKFold(n_splits=max(2, round(holdout / test_frac)),
                                 shuffle=True, random_state=seed)
    v_idx, t_idx = next(inner.split(hold, y=hold.label, groups=hold.patient))
    df.loc[hold.index[v_idx], "split"] = "val"
    df.loc[hold.index[t_idx], "split"] = "test"

    _assert_no_leakage(df)
    return df


def subsample(df: pd.DataFrame, n_train: int, seed: int = 42) -> pd.DataFrame:
    """Shrink the training split for a fast end-to-end rehearsal on real data.

    val and test are left intact so the numbers stay comparable to a full run.
    """
    if not n_train or n_train >= (df.split == "train").sum():
        return df
    train = df[df.split == "train"]
    keep = (train.groupby("label", group_keys=False)
                 .sample(frac=n_train / len(train), random_state=seed))
    return pd.concat([keep, df[df.split != "train"]]).sort_index()


def _assert_no_leakage(df: pd.DataFrame) -> None:
    groups = {s: set(g.patient) for s, g in df.groupby("split")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = groups.get(a, set()) & groups.get(b, set())
        if shared:
            raise AssertionError(
                f"patient leakage between {a} and {b}: {sorted(shared)[:5]} ... "
                f"({len(shared)} groups)")


def duplicate_images_across_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical images appearing in more than one split.

    Cheap integrity check: a duplicate spanning train and test would inflate the
    headline score, and no id-based check can see it.
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
    """Images per split and class, with the detailed RSNA class breakdown."""
    g = (df.groupby(["split", "cls"])
           .agg(images=("path", "size"), boxes=("n_boxes", "sum"))
           .reset_index())
    order = {"train": 0, "val": 1, "test": 2}
    return g.sort_values(["split", "cls"], key=lambda s: s.map(order).fillna(s))


def detailed_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.pivot_table(index="detailed_class", columns="split",
                           values="path", aggfunc="size", fill_value=0)
              .reindex(columns=["train", "val", "test"], fill_value=0))


def class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights computed on the training split only."""
    tr = df[df.split == "train"]
    n = len(tr)
    counts = tr.label.value_counts().to_dict()
    return {int(k): n / (2.0 * v) for k, v in counts.items()}


def scale_boxes(boxes_json: str, src_w: int, src_h: int, size: int) -> str:
    """Rescale [x, y, w, h] boxes from original pixels into the resized image."""
    boxes = json.loads(boxes_json) if isinstance(boxes_json, str) else (boxes_json or [])
    if not boxes:
        return "[]"
    sx, sy = size / float(src_w), size / float(src_h)
    return json.dumps([[x * sx, y * sy, w * sx, h * sy] for x, y, w, h in boxes])


def parse_boxes(value) -> np.ndarray:
    """splits.csv cell -> (N, 4) float array of [x, y, w, h]. Empty -> (0, 4)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.zeros((0, 4), dtype="float32")
    boxes = json.loads(value) if isinstance(value, str) else value
    if not boxes:
        return np.zeros((0, 4), dtype="float32")
    return np.asarray(boxes, dtype="float32").reshape(-1, 4)
