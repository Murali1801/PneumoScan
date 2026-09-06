"""Central configuration for the chest X-ray pneumonia screening pipeline."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Repository root = parent of src/
ROOT = Path(__file__).resolve().parent.parent

BACKBONES = ("densenet121", "efficientnetb0")
CLASSES = ("NORMAL", "PNEUMONIA")  # index 0, 1 -> label used for the sigmoid target


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    raw_dir: str = str(ROOT / "data" / "raw" / "chest_xray")
    processed_dir: str = str(ROOT / "data" / "processed")
    splits_csv: str = str(ROOT / "data" / "splits.csv")
    out_dir: str = str(ROOT / "reports")
    ckpt_dir: str = str(ROOT / "checkpoints")

    # ---- CLAHE preprocessing ----------------------------------------------
    img_size: int = 224
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # ---- splitting ---------------------------------------------------------
    val_frac: float = 0.10  # carved out of the official train folder, patient-grouped
    seed: int = 42

    # ---- model -------------------------------------------------------------
    backbone: str = "densenet121"
    dropout: float = 0.35

    # ---- training ----------------------------------------------------------
    batch_size: int = 32
    head_epochs: int = 5  # phase 1: frozen backbone, train the head only
    head_lr: float = 1e-3
    finetune_epochs: int = 20  # phase 2: unfreeze the backbone (BN kept frozen)
    finetune_lr: float = 1e-4
    unfreeze_from: float = 0.5  # unfreeze the last 50% of backbone layers
    patience: int = 6
    use_class_weights: bool = True

    # ---- augmentation (train only) ----------------------------------------
    aug_rotation: float = 0.03  # fraction of 2*pi  -> about +/- 11 degrees
    aug_translation: float = 0.06
    aug_zoom: float = 0.10
    aug_contrast: float = 0.10
    # Deliberately NO horizontal flip: mirroring a chest X-ray creates
    # anatomically impossible images (dextrocardia) and can teach the model
    # to ignore laterality cues.

    # ---- misc --------------------------------------------------------------
    tag: str = ""  # optional run name suffix

    # ---- derived -----------------------------------------------------------
    @property
    def run_name(self) -> str:
        return f"{self.backbone}{('_' + self.tag) if self.tag else ''}"

    @property
    def run_dir(self) -> Path:
        p = Path(self.out_dir) / self.run_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def model_path(self) -> Path:
        p = Path(self.ckpt_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{self.run_name}.keras"

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        d = Config()
        for name, value in asdict(d).items():
            if isinstance(value, bool):
                p.add_argument(f"--{name.replace('_', '-')}",
                               dest=name, default=value,
                               action=argparse.BooleanOptionalAction)
            else:
                p.add_argument(f"--{name.replace('_', '-')}",
                               dest=name, type=type(value), default=value)
        return p

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        fields = {f for f in asdict(cls()).keys()}
        return cls(**{k: v for k, v in vars(args).items() if k in fields})
