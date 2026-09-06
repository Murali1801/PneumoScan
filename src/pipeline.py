"""tf.data input pipeline over the CLAHE-processed PNGs."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter(cfg) -> keras.Sequential:
    """Geometric + photometric jitter appropriate for chest radiographs.

    No horizontal flip on purpose - see Config.aug_* for the reasoning.
    Rotation/translation/zoom fill with black, matching the X-ray background.
    """
    return keras.Sequential(
        [
            layers.RandomRotation(cfg.aug_rotation, fill_mode="constant",
                                  fill_value=0.0, name="rand_rot"),
            layers.RandomTranslation(cfg.aug_translation, cfg.aug_translation,
                                     fill_mode="constant", fill_value=0.0,
                                     name="rand_shift"),
            layers.RandomZoom(cfg.aug_zoom, fill_mode="constant",
                              fill_value=0.0, name="rand_zoom"),
            layers.RandomContrast(cfg.aug_contrast, name="rand_contrast"),
        ],
        name="augment",
    )


def _decode(path, label, img_size: int):
    """PNG on disk (already CLAHE'd + resized) -> uint8 (H, W, 1)."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (img_size, img_size), method="area")
    img = tf.cast(tf.round(img), tf.uint8)
    img.set_shape((img_size, img_size, 1))
    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels, cfg, *, training: bool,
                 shuffle_buffer: int | None = None,
                 cache: bool = True) -> tf.data.Dataset:
    """Build the input pipeline.

    Caching happens on the *grayscale uint8* tensors (about 50 KB/image) rather
    than on the float RGB batches, which keeps the whole training set in a few
    hundred MB of RAM instead of a few GB.  Augmentation runs after batching so
    the Keras random layers process a whole batch per call.
    """
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype="float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _decode(p, y, cfg.img_size), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(shuffle_buffer or min(len(paths), 4096),
                        seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.batch(cfg.batch_size, drop_remainder=False)

    augment = build_augmenter(cfg) if training else None

    def _to_float(x, y):
        x = tf.image.grayscale_to_rgb(x)
        x = tf.cast(x, tf.float32)          # [0, 255] - the model's input range
        if augment is not None:
            x = augment(x, training=True)
            x = tf.clip_by_value(x, 0.0, 255.0)
        return x, y

    ds = ds.map(_to_float, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def datasets_from_frame(df, cfg, splits=("train", "val", "test")) -> dict:
    """Convenience: split DataFrame (with a `proc_path` column) -> datasets."""
    out = {}
    for s in splits:
        sub = df[df.split == s]
        out[s] = make_dataset(sub.proc_path.tolist(), sub.label.tolist(),
                              cfg, training=(s == "train"))
    return out
