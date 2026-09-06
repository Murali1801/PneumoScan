"""Assemble a deployable copy of the demo for Hugging Face Spaces.

    python scripts/prepare_space.py                      # best checkpoint
    python scripts/prepare_space.py --run densenet121    # a specific one

Writes a self-contained folder (default `build/space/`) holding only what the
demo actually needs. Why a separate build rather than deploying the repo:

* **Dependencies.** The app imports five packages; the training pipeline needs
  pandas, scikit-learn, matplotlib and kaggle as well. Shipping the training
  requirements would install roughly a gigabyte the demo never touches, and free
  hosting tiers are disk- and memory-limited.
* **`tensorflow-cpu`, not `tensorflow`.** On Linux the default wheel drags in the
  CUDA runtime - hundreds of megabytes of GPU libraries that cannot be used on a
  CPU-only host.
* **Checkpoints are gitignored** in this repo on purpose, but the Space needs one.
  Copying it explicitly keeps that decision visible instead of accidental.

Grad-CAM needs gradients, so the Space runs the full Keras model rather than the
7 MB TFLite file - TFLite can do inference but cannot backpropagate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

# Only what the app imports. Deliberately not the training requirements.
REQUIREMENTS = """\
streamlit>=1.49
tensorflow-cpu==2.19.*
opencv-python-headless>=4.9
numpy>=1.26,<3
pydicom>=2.4
"""

# Hugging Face retired the Streamlit SDK - it now accepts only gradio, docker or
# static - so Streamlit apps ship as Docker Spaces. Two things Spaces requires
# that a plain Dockerfile would miss: it routes traffic to port 7860, and the
# container runs as uid 1000, so root-owned files are unreadable.
DOCKERFILE = """\
FROM python:3.11-slim

# opencv-python-headless still links against glib even without the GUI parts
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# requirements before code, so editing the app does not reinstall TensorFlow
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \\
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \\
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
"""

SPACE_README = """\
---
title: PneumoScan
emoji: 🫁
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PneumoScan — Explainable Pneumonia Screening

Upload a chest X-ray (DICOM, PNG or JPEG) and get a prediction with a Grad-CAM
heatmap showing what drove it.

**Research demonstration only — not a medical device.**

| | |
|---|---|
| Model | DenseNet121, {size}×{size} input, CLAHE preprocessing |
| Data | RSNA Pneumonia Detection Challenge (predominantly adult) |
| Test AUROC | {auroc} on {n} held-out images |
| Sensitivity / Specificity | {sens} / {spec} |
| NPV / PPV | {npv} / {ppv} |

The decision threshold is the value tuned on validation data, not 0.5 — at this
prevalence 0.5 is arbitrary. **NPV is the number that matters**: this is a
rule-out aid. A negative is trustworthy; a positive needs a radiologist.

Grad-CAM localisation was validated against the radiologist bounding boxes that
ship with RSNA — the heatmap's hottest pixel lands inside a box 5.4× more often
than chance.

Full training pipeline: {repo}
"""


def shrink_checkpoint(src: Path, dst: Path) -> tuple[float, float]:
    """Rewrite a .keras checkpoint with float16 weights, deflated.

    Three separate wins, none of which touch the app:

    * **No optimizer.** A checkpoint saved after training carries Adam's two
      moment estimates per weight - roughly double the file, for state inference
      never reads. `compile=False` drops them.
    * **float16 weights.** Weights are *stored* as float16 and Keras casts them
      back on load, so the graph still computes in float32. Measured drift on
      this model is ~1e-3 of a probability, with no decisions flipped at the
      operating threshold.
    * **Actual compression.** Keras writes the .keras zip with method 0, i.e.
      stored, not deflated.

    The HDF5 payload is rebuilt into a *new* file rather than edited in place:
    deleting a dataset leaves its bytes allocated, so an in-place edit produces a
    file that is still full size.

    Returns (size before, size after) in bytes.
    """
    import zipfile

    import h5py
    import numpy as np
    from tensorflow import keras

    before = src.stat().st_size
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "unpacked"
        stripped = Path(td) / "stripped.keras"
        keras.models.load_model(src, compile=False).save(stripped)
        with zipfile.ZipFile(stripped) as z:
            z.extractall(work)

        h5path = work / "model.weights.h5"
        newpath = work / "new.weights.h5"

        def copy_into(s, d):
            for k, a in s.attrs.items():
                d.attrs[k] = a
            for k, obj in s.items():
                if isinstance(obj, h5py.Group):
                    copy_into(obj, d.create_group(k))
                else:
                    data = obj[()]
                    if data.dtype == np.float32:
                        data = data.astype(np.float16)
                    ds = d.create_dataset(k, data=data)
                    for kk, aa in obj.attrs.items():
                        ds.attrs[kk] = aa

        with h5py.File(h5path, "r") as fin, h5py.File(newpath, "w") as fout:
            copy_into(fin, fout)
        h5path.unlink()
        newpath.rename(h5path)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(work.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(work).as_posix())
    return before, dst.stat().st_size


def verify_checkpoint(original: Path, shrunk: Path, size: int,
                      threshold: float, n: int = 24) -> float:
    """Both checkpoints must agree, and agree on the *decision*, not just the score."""
    import numpy as np
    from tensorflow import keras

    a_model = keras.models.load_model(original, compile=False)
    b_model = keras.models.load_model(shrunk, compile=False)
    x = (np.random.default_rng(0).random((n, size, size, 3)) * 255).astype("float32")
    a = a_model.predict(x, verbose=0).ravel()
    b = b_model.predict(x, verbose=0).ravel()
    flips = int(((a >= threshold) != (b >= threshold)).sum())
    drift = float(np.abs(a - b).max())
    if flips:
        raise SystemExit(f"float16 flipped {flips}/{n} decisions at threshold "
                         f"{threshold:.3f} - ship the float32 checkpoint instead "
                         "(--no-fp16)")
    return drift


def check_local_imports(out: Path) -> list[tuple[str, str]]:
    """Every project module the built files import must exist in the build.

    A missing one does not fail the build - it fails at runtime, on the Space,
    after a push. `app/discovery.py` was left out of the copy list exactly once
    and the folder looked perfectly healthy until the app was started. This walks
    the copied Python files, collects bare `import x` / `from x import ...` names,
    and reports any that name a project module but are not present.
    """
    import ast

    shipped = {p.stem for p in out.rglob("*.py")}
    available = {p.stem for p in (ROOT / "app").glob("*.py")}
    available |= {p.stem for p in (ROOT / "src").glob("*.py")}

    missing: list[tuple[str, str]] = []
    for path in sorted(out.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".")[0]] if node.module and not node.level else []
            else:
                continue
            for name in names:
                # only project modules matter; third-party ones come from pip
                if name in available and name not in shipped:
                    missing.append((path.relative_to(out).as_posix(), name))
    return sorted(set(missing))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="checkpoint stem to ship (default: highest test AUROC)")
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    ap.add_argument("--repo", default="https://github.com/Murali1801/pneumoscan",
                    help="link back to the full pipeline, shown in the Space README")
    ap.add_argument("--fp16", default=True, action=__import__("argparse").BooleanOptionalAction,
                    help="store weights as float16 (about 2.3x smaller, ~1e-3 drift)")
    args = ap.parse_args()

    import discovery

    models = discovery.catalogue()
    if not models:
        print("No checkpoint found. Copy a trained .keras into checkpoints/ first.")
        return 1
    entry = next((m for m in models if m["name"] == args.run), None) if args.run \
        else models[0]
    if entry is None:
        print(f"--run {args.run!r} not found. Available: "
              + ", ".join(m["name"] for m in models))
        return 1
    if entry["metrics"] is None:
        print(f"{entry['name']} has no metrics.json - the app would fall back to a "
              "0.5 threshold. Copy its report folder in first.")
        return 1

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "app").mkdir(parents=True)
    (out / "src").mkdir()
    (out / ".streamlit").mkdir()
    (out / "checkpoints").mkdir()
    (out / "reports" / entry["name"]).mkdir(parents=True)

    # Only the modules the app actually imports - src/data.py, train.py and the
    # rest belong to the training pipeline and would drag pandas and sklearn in.
    for f in ("app/streamlit_app.py", "app/theme.py", "app/discovery.py",
              "src/gradcam.py", "src/preprocess.py",
              ".streamlit/config.toml"):
        shutil.copy2(ROOT / f, out / f)

    dst_ckpt = out / "checkpoints" / entry["path"].name
    cfg = discovery.find_config(entry["name"]) or {}
    size = cfg.get("img_size", 224)

    if args.fp16:
        before, after = shrink_checkpoint(entry["path"], dst_ckpt)
        drift = verify_checkpoint(entry["path"], dst_ckpt, int(size),
                                  entry["threshold"])
        print(f"checkpoint {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB "
              f"({before / after:.1f}x: no optimizer, float16 weights, deflated)")
        print(f"  verified against the original: max drift {drift:.1e}, "
              "no decisions flipped")
    else:
        from tensorflow import keras

        before = entry["path"].stat().st_size
        keras.models.load_model(entry["path"], compile=False).save(dst_ckpt)
        print(f"checkpoint {before / 1e6:.1f} MB -> "
              f"{dst_ckpt.stat().st_size / 1e6:.1f} MB (optimizer removed)")
    src_metrics = next(d / entry["name"] / "metrics.json"
                       for d in discovery.REPORT_DIRS
                       if (d / entry["name"] / "metrics.json").is_file())
    shutil.copy2(src_metrics, out / "reports" / entry["name"] / "metrics.json")

    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")

    t = entry["metrics"]["test"]
    (out / "README.md").write_text(SPACE_README.format(
        size=size,
        auroc=f"{t['auroc']:.3f}",
        n=f"{sum(t['confusion_matrix'].values()):,}",
        sens=f"{t['sensitivity_recall']:.3f}",
        spec=f"{t['specificity']:.3f}",
        npv=f"{t['npv']:.3f}",
        ppv=f"{t['precision_ppv']:.3f}",
        repo=args.repo,
    ), encoding="utf-8")

    missing = check_local_imports(out)
    if missing:
        print("build is incomplete - these local modules are imported but absent:")
        for who, mod in missing:
            print(f"  {who} imports {mod!r}")
        print("\nAdd them to the copy list in this script.")
        return 1

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out).as_posix()
            print(f"  {rel:<44} {f.stat().st_size / 1024:>8,.0f} KB")
    print(f"\n  total {total / 1e6:.1f} MB  ·  model {entry['name']} "
          f"({size}×{size}, AUROC {t['auroc']:.3f}, threshold {t['threshold']:.3f})")
    print("\nnext:")
    print("  1. create a Space at https://huggingface.co/new-space (SDK: Docker)")
    print(f"  2. cd {out}")
    print("  3. git init && git lfs install && git lfs track '*.keras'")
    print("  4. git add -A && git commit -m 'PneumoScan demo'")
    print("  5. git remote add origin https://huggingface.co/spaces/<you>/pneumoscan")
    print("  6. git push -u origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
