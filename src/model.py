"""DenseNet121 transfer-learning classifier.

Why DenseNet121: CheXNet (Rajpurkar et al., 2017) established this architecture
for pneumonia detection on NIH ChestX-ray14, and RSNA is a re-annotated subset of
exactly that collection - so this is the reference architecture evaluated on data
from the collection it was validated on.

Input contract (the single most important detail here): the model takes RGB
float32 in **[0, 255]**.  DenseNet's own normalisation is baked in as the first
two layers, so the tf.data pipeline, the Grad-CAM code, the localisation metric
and the TFLite model all feed the network the same thing.

Two named layers are the Grad-CAM contract:
  * `feature_map` - the last convolutional feature map (7x7x1024 at 224x224 input)
  * `logits`      - the pre-sigmoid score, whose gradient Grad-CAM differentiates
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import layers

BACKBONE = "densenet121"

# ImageNet statistics used by DenseNet's "torch" preprocessing mode.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


def build_model(backbone: str = BACKBONE, img_size: int = 224,
                dropout: float = 0.35) -> keras.Model:
    if backbone != BACKBONE:
        raise ValueError(
            f"unknown backbone {backbone!r}; this project trains {BACKBONE!r} only")

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")  # [0, 255]

    # DenseNet expects torch-style normalisation; express it as real layers
    # rather than a Lambda so the graph stays serialisable and TFLite-safe.
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.Normalization(axis=-1, mean=_IMAGENET_MEAN,
                             variance=_IMAGENET_VAR, name="imagenet_norm")(x)

    base = keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False
    # training=False keeps BatchNorm in inference mode for the whole run - the
    # standard fine-tuning recipe for a modest dataset with batch size 32.
    feats = base(x, training=False)

    feats = layers.Activation("linear", name="feature_map")(feats)
    x = layers.GlobalAveragePooling2D(name="gap")(feats)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(1, name="logits")(x)
    prob = layers.Activation("sigmoid", name="prob")(logits)

    return keras.Model(inputs, prob, name=f"cxr_{backbone}")


def get_backbone(model: keras.Model) -> keras.Model:
    """The pretrained backbone is the only nested Model in the graph.

    Located by type rather than by name: `keras.applications` sets its own layer
    name, and renaming a built model is not reliable across Keras versions.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("no nested backbone model found in " + model.name)


def set_finetune(model: keras.Model, unfreeze_from: float = 0.5) -> int:
    """Unfreeze the deepest `unfreeze_from` fraction of backbone layers.

    Early layers hold generic edge/texture filters that transfer fine from
    ImageNet; the deeper blocks are the ones that need to re-specialise on
    radiographs.  BatchNorm layers stay frozen so their ImageNet moving
    statistics are not corrupted by small batches.
    """
    base = get_backbone(model)
    base.trainable = True
    n = len(base.layers)
    cut = int(n * (1.0 - unfreeze_from))

    trainable = 0
    for i, layer in enumerate(base.layers):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = i >= cut
        trainable += int(layer.trainable)
    return trainable


def compile_model(model: keras.Model, lr: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="auprc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def trainable_report(model: keras.Model) -> str:
    tr = sum(int(tf.size(w)) for w in model.trainable_weights)
    nt = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return f"trainable={tr:,}  frozen={nt:,}  total={tr + nt:,}"
