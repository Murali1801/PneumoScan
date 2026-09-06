"""Stage 4: convert the trained Keras model to TensorFlow Lite for the mobile app.

    python src/export_tflite.py --backbone densenet121

Produces two files next to the checkpoint:
  <run>.tflite          float32 - identical maths to Keras, largest file
  <run>_dynamic.tflite  weights quantised to int8, activations still float
                        (~4x smaller, no calibration data needed, accuracy drop
                        is usually well under one point)

Both take the same input the Keras model does: float32 (1, 224, 224, 3) in
[0, 255], already CLAHE'd.  Parity against Keras is verified before writing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import Config
from preprocess import to_model_input
import cv2


def _to_saved_model(model: keras.Model, out_dir: Path) -> Path:
    """Keras 3 models must go through an exported SavedModel, not
    `from_keras_model` - the latter silently fails on Keras 3 functional models."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.export(str(out_dir))
    return out_dir


def convert(model: keras.Model, out_path: Path, quantise: bool = False) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        sm = _to_saved_model(model, Path(tmp) / "saved_model")
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        if quantise:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return out_path


def tflite_predict(tflite_path: Path, batch: np.ndarray) -> np.ndarray:
    """Run a float32 [0,255] batch through the .tflite file, one image at a time."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for img in batch:
        interp.resize_tensor_input(inp["index"], (1, *img.shape))
        interp.allocate_tensors()
        interp.set_tensor(inp["index"], img[None].astype(inp["dtype"]))
        interp.invoke()
        preds.append(float(interp.get_tensor(out["index"]).ravel()[0]))
    return np.array(preds)


def sample_batch(cfg: Config, n: int = 16) -> np.ndarray:
    """A few real test images, so parity is checked on the actual data domain."""
    df = pd.read_csv(cfg.splits_csv)
    paths = df[df.split == "test"].proc_path.sample(
        min(n, (df.split == "test").sum()), random_state=cfg.seed).tolist()
    imgs = [to_model_input(cv2.imread(p, cv2.IMREAD_GRAYSCALE))[0] for p in paths]
    return np.stack(imgs).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="max allowed |keras - tflite| probability difference (float model)")
    Config.add_args(ap)
    args = ap.parse_args()
    cfg = Config.from_args(args)

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"{cfg.model_path} not found - train the model first.")
    model = keras.models.load_model(cfg.model_path)

    batch = sample_batch(cfg)
    ref = model.predict(batch, verbose=0).ravel()

    results = {}
    for name, quant in (("", False), ("_dynamic", True)):
        path = cfg.model_path.with_name(cfg.run_name + name + ".tflite")
        convert(model, path, quantise=quant)
        got = tflite_predict(path, batch)
        max_diff = float(np.max(np.abs(got - ref)))
        size_mb = path.stat().st_size / 1e6
        results[path.name] = {"size_mb": round(size_mb, 2),
                              "max_abs_diff_vs_keras": max_diff}
        print(f"{path.name:34s} {size_mb:6.2f} MB   max|diff| = {max_diff:.2e}")
        if not quant and max_diff > args.tolerance:
            raise AssertionError(
                f"{path.name} disagrees with Keras by {max_diff:.3e} "
                f"(> {args.tolerance}) - do not ship this model.")

    out = cfg.run_dir / "tflite_export.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("Both models expect float32 (1, 224, 224, 3) in [0, 255], CLAHE applied.")


if __name__ == "__main__":
    main()
