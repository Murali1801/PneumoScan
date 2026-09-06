"""Stage 2: two-phase transfer-learning on the CLAHE'd RSNA dataset.

    python src/train.py
    python src/train.py --finetune-epochs 20
    python src/train.py --subsample-train 3000 --tag rehearsal   # fast dry run

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
from data import class_weights, subsample
from evaluate import format_report, plot_history, run_evaluation
from model import (build_model, compile_model, set_finetune, to_float32,
                   trainable_report)
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


def _phase1_path(cfg: Config) -> Path:
    return cfg.model_path.with_name(cfg.run_name + "_phase1.keras")


def _epochs_done(cfg: Config, phase: str) -> int:
    """How many epochs of a phase already ran, read back from its CSV log.

    Reads the `epoch` column rather than counting lines: Keras's CSVLogger opens
    the file in text mode and lets csv.writer emit \\r\\n, so on Windows every
    row ends \\r\\r\\n and a naive line count comes out nearly double. Keras
    numbers epochs from 0, so the answer is last epoch + 1.
    """
    path = cfg.run_dir / f"history_{phase}.csv"
    if not path.exists():
        return 0
    import csv

    with open(path, encoding="utf-8", newline="") as fh:
        epochs = [int(row["epoch"]) for row in csv.DictReader(fh)
                  if str(row.get("epoch", "")).strip().isdigit()]
    return max(epochs) + 1 if epochs else 0


def _callbacks(cfg: Config, phase: str, best_so_far: float | None = None,
               append: bool = False):
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
        keras.callbacks.CSVLogger(str(log_dir / f"history_{phase}.csv"),
                                  append=append),
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
    if cfg.subsample_train:
        df = subsample(df, cfg.subsample_train, cfg.seed)
        print(f"rehearsal mode: training on {(df.split == 'train').sum():,} "
              "images; val and test left at full size")
    datasets = datasets_from_frame(df, cfg)
    cw = class_weights(df) if cfg.use_class_weights else None

    counts = df[df.split == "train"].cls.value_counts().to_dict()
    weights_str = {k: round(v, 3) for k, v in cw.items()} if cw else "off"
    print(f"train images: {counts}   class weights: {weights_str}")

    t0 = time.time()
    histories = []
    done_head = _epochs_done(cfg, "head") if cfg.resume else 0
    done_fine = _epochs_done(cfg, "finetune") if cfg.resume else 0
    into_phase2 = (cfg.resume and cfg.model_path.exists()
                   and done_head >= cfg.head_epochs and done_fine > 0)

    if into_phase2:
        print(f"resuming: phase 1 done, phase 2 at epoch "
              f"{done_fine}/{cfg.finetune_epochs}")
        model = keras.models.load_model(cfg.model_path)
        best_head = None
    else:
        done_fine = 0
        if cfg.resume and _phase1_path(cfg).exists():
            print("resuming: reloading the phase 1 checkpoint")
            model = keras.models.load_model(_phase1_path(cfg))
            best_head = None
        else:
            model = build_model(cfg.backbone, cfg.img_size, cfg.dropout)
            compile_model(model, cfg.head_lr)
            print(f"\n[phase 1] frozen backbone - {trainable_report(model)}")
            h1 = model.fit(datasets["train"], validation_data=datasets["val"],
                           epochs=cfg.head_epochs, class_weight=cw,
                           initial_epoch=done_head,
                           callbacks=_callbacks(cfg, "head", append=done_head > 0),
                           verbose=1)
            histories.append(h1)
            best_head = max(h1.history.get("val_auc", [0.0]))
            # Phase 1 is short but not free - keep it, so a phase 2 failure
            # never costs the warm-up as well.
            model.save(_phase1_path(cfg))

        n_unfrozen = set_finetune(model, cfg.unfreeze_from)
        compile_model(model, cfg.finetune_lr)  # recompile so the change takes effect
        print(f"\n[phase 2] fine-tuning {n_unfrozen} backbone layers - "
              f"{trainable_report(model)}")

    h2 = model.fit(datasets["train"], validation_data=datasets["val"],
                   epochs=cfg.finetune_epochs, class_weight=cw,
                   initial_epoch=done_fine,
                   callbacks=_callbacks(cfg, "finetune", best_head,
                                        append=done_fine > 0),
                   verbose=1)
    histories.append(h2)
    minutes = (time.time() - t0) / 60

    history = _merge_history(*histories)
    plot_history(history, cfg.run_dir / "training_curves.png")
    (cfg.run_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")
    cfg.save(cfg.run_dir / "config.json")

    # EarlyStopping only restores best weights when it actually fires, so reload
    # from the checkpoint - that file is always the best-val-AUROC epoch.
    if cfg.model_path.exists():
        model = keras.models.load_model(cfg.model_path)
        if history.get("val_auc"):
            print(f"reloaded best checkpoint (val AUROC "
                  f"{max(history['val_auc']):.4f})")

    # Strip the mixed-float16 policy out of the artefact everything else uses,
    # so Grad-CAM and the TFLite converter see an ordinary float32 graph.
    if cfg.mixed_precision:
        model = to_float32(model, cfg.img_size, cfg.dropout)
        compile_model(model, cfg.finetune_lr)
        model.save(cfg.model_path)
        print("saved a float32 copy of the best checkpoint")

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

    if cfg.mixed_precision:
        if not gpus:
            print("mixed precision needs a GPU - staying in float32")
            cfg.mixed_precision = False
        else:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("mixed precision ON (float16 compute, float32 weights)")
    for g in gpus:  # avoid grabbing all VRAM up front
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    train(cfg)


if __name__ == "__main__":
    main()
