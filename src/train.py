"""Stage 2: two-phase transfer-learning on the CLAHE'd Kermany dataset.

    python src/train.py --backbone densenet121
    python src/train.py --backbone efficientnetb0 --finetune-epochs 25

Phase 1 (warm-up)  - backbone frozen, only the new head trains.  Without this,
                     the large random gradients from an untrained head would
                     wreck the pretrained ImageNet filters in the first batches.
Phase 2 (fine-tune)- the deepest half of the backbone unfreezes at a 10x lower
                     learning rate; BatchNorm stays in inference mode.

Model selection is on validation AUROC, never on the test set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import Config
from data import class_weights
from evaluate import format_report, plot_history, run_evaluation
from model import build_model, compile_model, set_finetune, trainable_report
from pipeline import datasets_from_frame


def load_splits(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.splits_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/prepare_data.py` first.")
    df = pd.read_csv(path)
    missing = [p for p in df.proc_path.head(20) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "processed images referenced by splits.csv are missing, e.g.\n  "
            + "\n  ".join(missing[:3])
            + "\nRe-run scripts/prepare_data.py (paths are absolute in the CSV).")
    return df


def _callbacks(cfg: Config, phase: str, best_so_far: float | None = None):
    log_dir = cfg.run_dir
    return [
        # `initial_value_threshold` carries phase 1's best val AUROC into phase 2
        # so the checkpoint is never overwritten by a worse fine-tuning epoch.
        keras.callbacks.ModelCheckpoint(
            str(cfg.model_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
            initial_value_threshold=best_so_far),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=cfg.patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.3,
            patience=max(2, cfg.patience // 2), min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(log_dir / f"history_{phase}.csv")),
    ]


def _merge_history(*histories) -> dict:
    out: dict[str, list] = {}
    for h in histories:
        for k, v in h.history.items():
            out.setdefault(k, []).extend([float(x) for x in v])
    return out


def train(cfg: Config) -> dict:
    keras.utils.set_random_seed(cfg.seed)

    df = load_splits(cfg)
    datasets = datasets_from_frame(df, cfg)
    cw = class_weights(df) if cfg.use_class_weights else None

    counts = df[df.split == "train"].cls.value_counts().to_dict()
    weights_str = {k: round(v, 3) for k, v in cw.items()} if cw else "off"
    print(f"train images: {counts}   class weights: {weights_str}")

    model = build_model(cfg.backbone, cfg.img_size, cfg.dropout)
    compile_model(model, cfg.head_lr)
    print(f"\n[phase 1] frozen backbone - {trainable_report(model)}")
    t0 = time.time()
    h1 = model.fit(datasets["train"], validation_data=datasets["val"],
                   epochs=cfg.head_epochs, class_weight=cw,
                   callbacks=_callbacks(cfg, "head"), verbose=1)
    best_head = max(h1.history.get("val_auc", [0.0]))

    n_unfrozen = set_finetune(model, cfg.unfreeze_from)
    compile_model(model, cfg.finetune_lr)  # recompile so the change takes effect
    print(f"\n[phase 2] fine-tuning {n_unfrozen} backbone layers - "
          f"{trainable_report(model)}")
    h2 = model.fit(datasets["train"], validation_data=datasets["val"],
                   epochs=cfg.finetune_epochs, class_weight=cw,
                   callbacks=_callbacks(cfg, "finetune", best_head), verbose=1)
    minutes = (time.time() - t0) / 60

    history = _merge_history(h1, h2)
    plot_history(history, cfg.run_dir / "training_curves.png")
    (cfg.run_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")
    cfg.save(cfg.run_dir / "config.json")

    # EarlyStopping only restores best weights when it actually fires, so reload
    # from the checkpoint - that file is always the best-val-AUROC epoch.
    if cfg.model_path.exists():
        model = keras.models.load_model(cfg.model_path)
        print(f"reloaded best checkpoint (val AUROC "
              f"{max(history.get('val_auc', [float('nan')])):.4f})")

    print(f"\nevaluating {cfg.run_name} ...")
    report = run_evaluation(model, datasets, cfg.run_dir)
    report["train_minutes"] = round(minutes, 2)
    report["backbone"] = cfg.backbone
    (cfg.run_dir / "metrics.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")

    print()
    print(format_report(report, cfg.run_name))
    print(f"\n  trained in {minutes:.1f} min")
    print(f"  model    -> {cfg.model_path}")
    print(f"  artefacts-> {cfg.run_dir}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    Config.add_args(ap)
    cfg = Config.from_args(ap.parse_args())

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | "
          f"GPU: {[g.name for g in gpus] or 'none (CPU - this will be slow)'}")
    for g in gpus:  # avoid grabbing all VRAM up front
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    train(cfg)


if __name__ == "__main__":
    main()
