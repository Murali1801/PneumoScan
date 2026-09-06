"""CLAHE preprocessing.

The SAME `clahe_image` function is used for the offline dataset pass, for the
Grad-CAM figures and for inference in the demo app.  Keeping one implementation
is what guarantees that what the model saw during training is exactly what it
sees at prediction time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def clahe_image(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on a uint8 grayscale image.

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


def read_gray(path: str | Path) -> np.ndarray:
    """Read any image as uint8 grayscale, raising a clear error on bad files."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:  # cv2 fails silently on unicode paths / corrupt files
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"could not decode image: {path}")
    return img


def prepare_image(path: str | Path, size: int = 224,
                  clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Full offline path: read -> CLAHE -> resize.  Returns uint8 (size, size)."""
    img = read_gray(path)
    img = clahe_image(img, clip, grid)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(gray_224: np.ndarray) -> np.ndarray:
    """uint8 (H, W) -> float32 (1, H, W, 3) in [0, 255], the model's input range."""
    rgb = cv2.cvtColor(gray_224, cv2.COLOR_GRAY2RGB).astype("float32")
    return rgb[None, ...]


def preprocess_dataset(rows, out_root: str | Path, size: int = 224,
                       clip: float = 2.0, grid: int = 8, workers: int = 8) -> list[str]:
    """Run `prepare_image` over an iterable of (src_path, rel_out_path) pairs.

    Writes lossless PNGs so the CLAHE result is not re-degraded by JPEG.
    Returns the list of written paths, in input order.
    """
    out_root = Path(out_root)
    rows = list(rows)

    def _one(item):
        src, rel = item
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            cv2.imwrite(str(dst), prepare_image(src, size, clip, grid))
        return str(dst)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, rows))
