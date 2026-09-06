"""Prove the whole pipeline runs before spending an hour on the real data.

    python scripts/selftest.py

Builds a small synthetic RSNA-shaped dataset - genuine DICOMs plus the two
competition CSVs - then runs every stage end to end:

    prepare_data -> train -> Grad-CAM + localisation -> TFLite export

and checks the expected artefacts actually appeared.  Takes ~2 minutes on a GPU,
~5 on CPU, and touches nothing outside a temporary directory.

The synthetic DICOMs are deliberately **512x512, not 1024**, so that rescaling
the bounding boxes into the model's frame has to genuinely work rather than pass
by accident on a 1:1 scale factor.

Exit code 0 means every stage ran and produced its outputs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

SIZE = 512
SC_UID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
NOT_NORMAL = "No Lung Opacity / Not Normal"

rng = np.random.default_rng(11)


# --------------------------------------------------------------------------
# synthetic dataset
# --------------------------------------------------------------------------
def write_dicom(path: Path, arr: np.ndarray, age: int, sex: str, view: str) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SC_UID
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SC_UID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CR"
    ds.PatientID = path.stem
    ds.PatientName = "ANON"
    ds.PatientAge = f"{age:03d}Y"
    ds.PatientSex = sex
    ds.ViewPosition = view
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)      # pydicom >= 3
    except TypeError:                                        # pydicom 2.x
        ds.save_as(str(path), write_like_original=False)


def synth(kind: str):
    """Crude thorax phantom. Returns (uint8 image, [[x, y, w, h], ...])."""
    img = np.zeros((SIZE, SIZE), np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    body = ((xx - SIZE / 2) ** 2 / (SIZE * 0.40) ** 2
            + (yy - SIZE / 2) ** 2 / (SIZE * 0.46) ** 2) < 1
    img[body] = 130
    for cx in (0.32, 0.68):
        lung = ((xx - SIZE * cx) ** 2 / (SIZE * 0.16) ** 2
                + (yy - SIZE * 0.52) ** 2 / (SIZE * 0.27) ** 2) < 1
        img[lung] = 55
    img[:, int(SIZE * 0.46):int(SIZE * 0.54)] = 165

    boxes = []
    if kind == "opacity":
        for _ in range(int(rng.integers(1, 3))):
            cx = int(SIZE * rng.choice([0.32, 0.68]) + rng.normal(0, SIZE * 0.04))
            cy = int(SIZE * 0.52 + rng.normal(0, SIZE * 0.09))
            r = int(rng.integers(SIZE // 13, SIZE // 7))
            blob = np.zeros_like(img)
            cv2.circle(blob, (cx, cy), r, float(rng.uniform(60, 100)), -1)
            img += cv2.GaussianBlur(blob, (0, 0), r / 2.5)
            x0, y0 = max(0, cx - r), max(0, cy - r)
            boxes.append([x0, y0, min(2 * r, SIZE - x0), min(2 * r, SIZE - y0)])
    elif kind == "not_normal":
        # abnormal, but not a lung-field opacity
        pts = np.array([[int(SIZE * .18), int(SIZE * .80)],
                        [int(SIZE * .42), int(SIZE * .80)],
                        [int(SIZE * .18), int(SIZE * .68)]], np.int32)
        cv2.fillPoly(img, [pts], 120.0)

    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype("uint8"), boxes


def build_dataset(raw: Path) -> int:
    img_dir = raw / "stage_2_train_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    plan = [("opacity", 80, "Lung Opacity"),
            ("normal", 60, "Normal"),
            ("not_normal", 60, NOT_NORMAL)]
    label_rows, class_rows = [], []
    for kind, n, cls_name in plan:
        for _ in range(n):
            pid = str(uuid.uuid4())
            img, boxes = synth(kind)
            write_dicom(img_dir / f"{pid}.dcm", img,
                        age=int(rng.integers(19, 92)),
                        sex=str(rng.choice(["M", "F"])),
                        view=str(rng.choice(["PA", "AP"])))
            if boxes:
                for x, y, w, h in boxes:
                    label_rows.append(dict(patientId=pid, x=x, y=y,
                                           width=w, height=h, Target=1))
                    class_rows.append({"patientId": pid, "class": cls_name})
            else:
                label_rows.append(dict(patientId=pid, x=np.nan, y=np.nan,
                                       width=np.nan, height=np.nan, Target=0))
                class_rows.append({"patientId": pid, "class": cls_name})

    pd.DataFrame(label_rows).to_csv(raw / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(class_rows).to_csv(raw / "stage_2_detailed_class_info.csv", index=False)
    return len(list(img_dir.glob("*.dcm")))


# --------------------------------------------------------------------------
# stage runner
# --------------------------------------------------------------------------
class Stage:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.failures: list[str] = []

    def run(self, name: str, cmd: list[str], expect: list[Path]) -> bool:
        print(f"\n[{name}]", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if self.verbose:
            print((proc.stdout or "").rstrip())

        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
            print(f"  FAILED (exit {proc.returncode})\n{tail}")
            self.failures.append(name)
            return False

        missing = [p for p in expect if not p.exists()]
        if missing:
            print("  ran, but did not produce: "
                  + ", ".join(m.name for m in missing))
            self.failures.append(f"{name} (missing outputs)")
            return False

        print("  OK  -> " + ", ".join(p.name for p in expect))
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show each stage's full output")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary directory instead of deleting it")
    args = ap.parse_args()

    try:
        import pydicom  # noqa: F401
    except ImportError:
        print("pydicom is not installed:  pip install pydicom")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="pneumoscan_selftest_"))
    raw = tmp / "raw"
    print(f"workspace: {tmp}")
    print("building a synthetic RSNA dataset (512x512 DICOMs) ...", flush=True)
    n = build_dataset(raw)
    print(f"  {n} DICOMs + 2 competition CSVs")

    common = ["--raw-dir", str(raw),
              "--processed-dir", str(tmp / "processed"),
              "--splits-csv", str(tmp / "splits.csv"),
              "--out-dir", str(tmp / "reports"),
              "--ckpt-dir", str(tmp / "checkpoints"),
              "--img-size", "96"]
    run_dir = tmp / "reports" / "densenet121"
    py = sys.executable
    s = Stage(args.verbose)

    ok = s.run("1/4  prepare_data (DICOM -> CLAHE -> splits -> boxes)",
               [py, "scripts/prepare_data.py", *common],
               [tmp / "splits.csv",
                tmp / "reports" / "dataset_summary.csv",
                tmp / "reports" / "cohort.csv",
                tmp / "reports" / "clahe_examples.png"])

    if ok:
        ok = s.run("2/4  train (2 epochs)",
                   [py, "src/train.py", *common, "--head-epochs", "1",
                    "--finetune-epochs", "1", "--batch-size", "16"],
                   [tmp / "checkpoints" / "densenet121.keras",
                    run_dir / "metrics.json",
                    run_dir / "training_curves.png",
                    run_dir / "test_curves.png"])

    if ok:
        ok = s.run("3/4  Grad-CAM + localisation metric",
                   [py, "scripts/make_gradcam_figures.py", *common, "--n", "3"],
                   [run_dir / "gradcam_pneumonia.png",
                    run_dir / "gradcam_vs_pp.png",
                    run_dir / "localization.json",
                    run_dir / "localization.csv"])

    if ok:
        ok = s.run("4/4  TFLite export (with Keras-parity check)",
                   [py, "src/export_tflite.py", *common],
                   [tmp / "checkpoints" / "densenet121.tflite",
                    tmp / "checkpoints" / "densenet121_dynamic.tflite",
                    run_dir / "tflite_export.json"])

    print("\n" + "=" * 62)
    if s.failures:
        print("SELF-TEST FAILED at: " + "; ".join(s.failures))
        print(f"workspace kept for inspection: {tmp}")
        return 1

    print("SELF-TEST PASSED - all four stages ran and produced their outputs.")
    if args.keep:
        print(f"workspace kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
