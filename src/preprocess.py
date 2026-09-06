"""Image loading and CLAHE preprocessing.

RSNA ships 1024x1024 DICOMs, so reading is a little more involved than opening a
JPEG: pixel data can be 8- or 16-bit, and MONOCHROME1 studies store an inverted
greyscale ramp that has to be flipped or every X-ray comes out as a photographic
negative.

The SAME `clahe_image` runs for the offline dataset pass, for the Grad-CAM
figures and for inference in the demo app - one implementation, so there is no
train/serve skew.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DICOM_EXT = {".dcm", ".dicom"}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_dicom(path: str | Path) -> np.ndarray:
    """DICOM -> uint8 greyscale, photometric interpretation honoured."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array

    # MONOCHROME1 = 0 is white. Invert so bone is bright in every image.
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        arr = arr.max() - arr

    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return arr


def parse_dicom_age(raw) -> float | None:
    """DICOM PatientAge -> age in years.

    The standard says the value is 4 characters, `nnnD`/`nnnW`/`nnnM`/`nnnY`
    (e.g. "058Y").  Plenty of real archives - RSNA included - just store a bare
    number of years instead ("51"), so a parser that insists on the unit suffix
    silently returns nothing for the entire dataset.  Both forms are accepted.
    """
    s = str(raw or "").strip().upper()
    if not s:
        return None
    if s[-1].isalpha():
        unit, digits = s[-1], s[:-1]
    else:
        unit, digits = "Y", s
    digits = digits.lstrip("0") or "0"
    if not digits.isdigit():
        return None
    n = int(digits)
    return {"Y": float(n), "M": n / 12.0, "W": n / 52.0, "D": n / 365.25}.get(unit)


def dicom_metadata(path: str | Path) -> dict:
    """Age / sex / view, for the cohort table in the report.

    `age_raw` is carried through so a parsing failure is diagnosable from the
    output rather than showing up as a column of blanks.
    """
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    raw_age = str(getattr(ds, "PatientAge", "") or "").strip()
    return {
        "age": parse_dicom_age(raw_age),
        "age_raw": raw_age or None,
        "sex": str(getattr(ds, "PatientSex", "") or "").strip() or None,
        "view": str(getattr(ds, "ViewPosition", "") or "").strip() or None,
        "rows": int(getattr(ds, "Rows", 0)) or None,
        "cols": int(getattr(ds, "Columns", 0)) or None,
    }


def read_gray(path: str | Path) -> np.ndarray:
    """Read any supported image as uint8 greyscale."""
    path = Path(path)
    if path.suffix.lower() in DICOM_EXT:
        return read_dicom(path)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:  # cv2 fails silently on unicode paths / corrupt files
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"could not decode image: {path}")
    return img


# --------------------------------------------------------------------------
# CLAHE
# --------------------------------------------------------------------------
def clahe_image(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on uint8 greyscale.

    Ordinary histogram equalisation stretches the global histogram and tends to
    blow out the mediastinum while flattening the lung fields.  CLAHE equalises
    inside small tiles and clips the histogram before redistributing it, so
    subtle consolidation and infiltrates in the lung fields become visible
    without amplifying sensor noise.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return op.apply(gray)


def prepare_image(path: str | Path, size: int = 224,
                  clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Full offline path: read -> CLAHE -> resize.  Returns uint8 (size, size)."""
    img = read_gray(path)
    img = clahe_image(img, clip, grid)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """uint8 (H, W) -> float32 (1, H, W, 3) in [0, 255], the model's input range."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype("float32")
    return rgb[None, ...]


# --------------------------------------------------------------------------
# batch pass
# --------------------------------------------------------------------------
MANIFEST = "_preprocess.json"


def check_manifest(out_root: str | Path, size: int, clip: float, grid: int) -> None:
    """Refuse to mix preprocessing settings inside one processed/ directory.

    Files are skipped when they already exist, which makes an interrupted run
    resumable - but it also means that re-running with a different --img-size
    would silently keep the old images at the old size, and the pipeline would
    then upsample 224px data to 384 and report the result as a 384px model.
    Recording the settings and comparing turns that silent corruption into an
    error that says exactly what to do.
    """
    import json

    out_root = Path(out_root)
    want = {"img_size": int(size), "clahe_clip": float(clip), "clahe_grid": int(grid)}
    path = out_root / MANIFEST

    if path.exists():
        have = json.loads(path.read_text(encoding="utf-8"))
        if have != want:
            diffs = [f"{k}: {have.get(k)} -> {want[k]}"
                     for k in want if have.get(k) != want[k]]
            raise SystemExit(
                f"\n{out_root} was built with different preprocessing:\n  "
                + "\n  ".join(diffs)
                + "\n\nThose images cannot be reused. Either:\n"
                f"  rm -rf {out_root}          # rebuild at the new settings\n"
                f"  --processed-dir <other>    # keep both side by side\n")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(want, indent=2), encoding="utf-8")


def preprocess_dataset(rows, out_root: str | Path, size: int = 224,
                       clip: float = 2.0, grid: int = 8, workers: int = 8,
                       progress_every: int = 2000) -> list[str]:
    """Run `prepare_image` over an iterable of (src_path, rel_out_path) pairs.

    Writes lossless PNGs so the CLAHE result is not re-degraded by JPEG.
    Returns the written paths, in input order.  Already-written files are
    skipped, so an interrupted run can simply be restarted.
    """
    out_root = Path(out_root)
    check_manifest(out_root, size, clip, grid)
    rows = list(rows)
    total = len(rows)
    done = 0

    def _one(item):
        nonlocal done
        src, rel = item
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            cv2.imwrite(str(dst), prepare_image(src, size, clip, grid))
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"    {done:,}/{total:,} images", flush=True)
        return str(dst)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, rows))
