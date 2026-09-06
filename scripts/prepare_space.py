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

SPACE_README = """\
---
title: PneumoScan
emoji: 🫁
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.49.0
app_file: app/streamlit_app.py
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

    shutil.copy2(entry["path"], out / "checkpoints" / entry["path"].name)
    src_metrics = next(d / entry["name"] / "metrics.json"
                       for d in discovery.REPORT_DIRS
                       if (d / entry["name"] / "metrics.json").is_file())
    shutil.copy2(src_metrics, out / "reports" / entry["name"] / "metrics.json")

    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    # Input size comes from the run's config.json rather than by loading the
    # model - no reason to spin up TensorFlow just to write a README.
    cfg = discovery.find_config(entry["name"]) or {}
    size = cfg.get("img_size", "?")

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
    print("  1. create a Space at https://huggingface.co/new-space (SDK: Streamlit)")
    print(f"  2. cd {out}")
    print("  3. git init && git lfs install && git lfs track '*.keras'")
    print("  4. git add -A && git commit -m 'PneumoScan demo'")
    print("  5. git remote add origin https://huggingface.co/spaces/<you>/pneumoscan")
    print("  6. git push -u origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
