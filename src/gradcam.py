"""Grad-CAM and Grad-CAM++ explanations for the pneumonia classifier.

What the heatmap actually is
----------------------------
Grad-CAM takes the last convolutional feature map A (7x7x1024 for DenseNet121)
and the scalar score y the network produced, then weights each channel k by how
strongly it pushed that score up:

    w_k = mean over the 7x7 map of  dy / dA_k          (importance of channel k)
    CAM = ReLU( sum_k w_k * A_k )                      (keep only positive evidence)

ReLU keeps regions that *support* the prediction and discards the ones arguing
against it.  The 7x7 map is then upsampled to 224x224 and colour-mapped.

Two deliberate choices:
  * The score differentiated is the **logit**, not the sigmoid probability.  A
    confident sigmoid saturates at ~1.0 where its gradient is ~0, which washes
    the heatmap out; the logit has no such plateau.
  * `target="predicted"` flips the sign for a NORMAL prediction, so the map then
    highlights evidence *for normal* rather than evidence for pneumonia.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

FEATURE_LAYER = "feature_map"
LOGIT_LAYER = "logits"


class CAMExplainer:
    """Grad-CAM / Grad-CAM++ over a model built by `src.model.build_model`."""

    def __init__(self, model: keras.Model, mode: str = "gradcam"):
        if mode not in ("gradcam", "gradcam++"):
            raise ValueError("mode must be 'gradcam' or 'gradcam++'")
        self.mode = mode
        self.model = model
        self.grad_model = keras.Model(
            model.inputs,
            [model.get_layer(FEATURE_LAYER).output, model.get_layer(LOGIT_LAYER).output],
            name="grad_model",
        )

    # -- core ---------------------------------------------------------------
    def explain(self, batch: np.ndarray, target: str = "pneumonia"):
        """batch: float32 (B, H, W, 3) in [0, 255].

        Returns (heatmaps (B, h, w) in [0, 1], probabilities (B,)).
        """
        x = tf.convert_to_tensor(batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv, logit = self.grad_model(x, training=False)
            tape.watch(conv)
            score = logit[:, 0]
            if target == "predicted":
                # +logit where the model says pneumonia, -logit where it says normal
                sign = tf.where(score > 0, 1.0, -1.0)
                score = score * sign
            elif target != "pneumonia":
                raise ValueError("target must be 'pneumonia' or 'predicted'")

        grads = tape.gradient(score, conv)  # (B, h, w, C)

        if self.mode == "gradcam":
            weights = tf.reduce_mean(grads, axis=(1, 2))  # (B, C)
        else:
            weights = self._gradcam_pp_weights(conv, grads)

        cam = tf.einsum("bhwc,bc->bhw", conv, weights)
        cam = tf.nn.relu(cam)
        cam = self._normalise(cam).numpy()
        probs = tf.sigmoid(logit[:, 0]).numpy()
        return cam, probs

    @staticmethod
    def _gradcam_pp_weights(conv, grads):
        """Grad-CAM++ pixel-wise alpha weights (Chattopadhyay et al., 2018)."""
        g2, g3 = grads ** 2, grads ** 3
        conv_sum = tf.reduce_sum(conv, axis=(1, 2), keepdims=True)
        denom = 2.0 * g2 + conv_sum * g3
        alpha = tf.math.divide_no_nan(g2, denom)
        return tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))

    @staticmethod
    def _normalise(cam):
        lo = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
        hi = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
        return tf.math.divide_no_nan(cam - lo, hi - lo)


# -- rendering --------------------------------------------------------------
def overlay(gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40,
            colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a [0,1] heatmap over a uint8 grayscale X-ray -> RGB uint8."""
    h, w = gray.shape[:2]
    hm = cv2.resize(heatmap.astype("float32"), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, 1.0)
    colour = cv2.applyColorMap((hm * 255).astype("uint8"), colormap)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1.0 - alpha, colour, alpha, 0.0)


DEFAULT_HEADERS = ("Original X-ray", "After CLAHE", "Grad-CAM overlay")


def figure_grid(items, out_path: str | Path, threshold: float = 0.5,
                title: str = "Grad-CAM explanations",
                headers: tuple[str, str, str] = DEFAULT_HEADERS):
    """Save an N x 3 panel of explanation cases.

    `items` is a list of dicts with keys:
        original (uint8 HxW), clahe (uint8 HxW or HxWx3), overlay (uint8 HxWx3),
        prob (float), truth (str), name (str)
    The three image keys map onto the three `headers` columns in that order, so
    the same helper renders both the original/CLAHE/Grad-CAM panel and the
    Grad-CAM vs Grad-CAM++ comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.4 * n))
    axes = np.atleast_2d(axes)

    for r, it in enumerate(items):
        pred = "PNEUMONIA" if it["prob"] >= threshold else "NORMAL"
        conf = it["prob"] if pred == "PNEUMONIA" else 1.0 - it["prob"]
        ok = (pred == it["truth"])
        for c, (img, head) in enumerate(zip(
                (it["original"], it["clahe"], it["overlay"]), headers)):
            ax = axes[r, c]
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(head, fontsize=11, pad=8)
        axes[r, 0].set_ylabel(f"{it['name']}\ntruth: {it['truth']}",
                              fontsize=8, rotation=0, ha="right", va="center", labelpad=45)
        axes[r, 2].set_xlabel(
            f"pred: {pred}  ({conf:.1%})  {'OK' if ok else 'MISS'}",
            fontsize=9, color=("#1b7f3b" if ok else "#c0392b"))

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
