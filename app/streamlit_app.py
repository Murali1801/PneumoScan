"""PneumoScan demo: upload a chest X-ray, get a prediction and a Grad-CAM heatmap.

    streamlit run app/streamlit_app.py

Everything the model needs is derived from the checkpoint itself:

* **Input size** is read from the model's input shape, so a 224px and a 320px
  checkpoint both work without changing a line here. Hard-coding it is exactly
  how a demo ends up silently feeding the network upsampled garbage.
* **CLAHE** uses `preprocess.clahe_image`, the same function that built the
  training set - one implementation, so there is no train/serve skew.
* **The decision threshold** comes from the run's `metrics.json`, where it was
  tuned on validation data. Defaulting to 0.5 would quietly change the operating
  point the reported sensitivity and specificity belong to.

Presentation lives in `app/theme.py`; this file stays about the model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import cv2
import numpy as np
import streamlit as st

import theme
from discovery import catalogue

st.set_page_config(page_title="PneumoScan", page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(theme.CSS, unsafe_allow_html=True)

ACCEPTED = ["dcm", "dicom", "png", "jpg", "jpeg", "bmp"]


@st.cache_resource(show_spinner="Loading model …")
def load_model(path_str: str):
    from tensorflow import keras
    from gradcam import CAMExplainer

    model = keras.models.load_model(path_str)
    size = int(model.input_shape[1])          # never hard-code this
    return model, size, {m: CAMExplainer(model, m)
                         for m in ("gradcam", "gradcam++")}


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def analyse(model, explainer, raw_gray: np.ndarray, size: int,
            clip: float, grid: int, threshold: float) -> dict:
    """raw uint8 greyscale -> prediction + heatmap. No Streamlit calls in here,
    so the whole path is testable without a browser."""
    from gradcam import overlay
    from preprocess import clahe_image, to_model_input

    clahe = cv2.resize(clahe_image(raw_gray, clip, grid), (size, size),
                       interpolation=cv2.INTER_AREA)
    heat, probs = explainer.explain(to_model_input(clahe), target="predicted")
    prob = float(probs[0])
    positive = prob >= threshold
    return {
        "prob": prob,
        "positive": positive,
        "label": "PNEUMONIA" if positive else "NORMAL",
        "confidence": prob if positive else 1.0 - prob,
        "clahe": clahe,
        "original": cv2.resize(raw_gray, (size, size), interpolation=cv2.INTER_AREA),
        "overlay": overlay(clahe, heat[0]),
    }


def read_upload(upload) -> np.ndarray:
    """Uploaded bytes -> uint8 greyscale, DICOM or ordinary image."""
    data = upload.getvalue()
    if Path(upload.name).suffix.lower() in (".dcm", ".dicom"):
        import io

        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            arr = arr.max() - arr
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return arr

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not decode {upload.name}")
    return img


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown(theme.hero(), unsafe_allow_html=True)

models = catalogue()
if not models:
    st.error("No `.keras` checkpoint found in `checkpoints/` or "
             "`results/checkpoints/`. Train a model first, or copy a trained "
             "checkpoint into `checkpoints/`.")
    st.stop()

with st.sidebar:
    st.markdown(theme.side_header("Model"), unsafe_allow_html=True)
    # The name alone: the sidebar is too narrow for a name plus a metric, and a
    # truncated "AUROC 0.8" is worse than none. The score goes underneath.
    entry = st.selectbox("Checkpoint", models, index=0,
                         label_visibility="collapsed",
                         format_func=lambda r: r["name"])
    choice, tuned, metrics = entry["path"], entry["threshold"], entry["metrics"]
    model, size, explainers = load_model(str(choice))
    auroc = (f'AUROC <b>{entry["auroc"]:.3f}</b> &middot; ' if entry["auroc"] else "")
    st.markdown(
        f'<div class="side-note">{auroc}input <b>{size}×{size}</b>, read from '
        "the model." + (" Best first." if len(models) > 1 else "") + "</div>",
        unsafe_allow_html=True)

    st.markdown(theme.side_header("Decision threshold"), unsafe_allow_html=True)
    threshold = st.slider("Flag above", 0.05, 0.95, float(tuned), 0.01,
                          label_visibility="collapsed")
    if abs(threshold - tuned) > 1e-9:
        st.markdown(
            f'<div class="side-note">Tuned value is <b>{tuned:.3f}</b>. The '
            "reported sensitivity and specificity belong to that point.</div>",
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-note">Tuned on validation with '
                    "Youden&rsquo;s J. Lower it to catch more pneumonia at the "
                    "cost of more false alarms.</div>", unsafe_allow_html=True)

    st.markdown(theme.side_header("Explanation"), unsafe_allow_html=True)
    method = st.radio("Method", ["gradcam", "gradcam++"], label_visibility="collapsed",
                      format_func=lambda m: "Grad-CAM" if m == "gradcam" else "Grad-CAM++")

    st.markdown(theme.side_header("Preprocessing"), unsafe_allow_html=True)
    with st.expander("CLAHE settings"):
        clip = st.slider("Clip limit", 0.5, 5.0, 2.0, 0.5)
        grid = st.slider("Tile grid", 4, 16, 8, 2)
        st.markdown('<div class="side-note">Defaults match training. Changing them '
                    "means the model sees images unlike the ones it learned from."
                    "</div>", unsafe_allow_html=True)

upload = st.file_uploader("Upload a chest X-ray  ·  DICOM, PNG or JPEG",
                          type=ACCEPTED)

# ---- idle state: show what the model is worth before it is asked anything ----
if upload is None:
    if metrics:
        t = metrics["test"]
        cm = t["confusion_matrix"]
        n = sum(cm.values())
        st.markdown(theme.eyebrow("Measured on held-out data"), unsafe_allow_html=True)
        st.markdown(theme.stats([
            ("AUROC", f"{t['auroc']:.3f}", "ranking quality"),
            ("Sensitivity", f"{t['sensitivity_recall']:.3f}", "pneumonia caught"),
            ("Specificity", f"{t['specificity']:.3f}", "normals cleared"),
            ("NPV", f"{t['npv']:.3f}", "negatives correct"),
            ("PPV", f"{t['precision_ppv']:.3f}", "positives correct"),
        ], lead="NPV"), unsafe_allow_html=True)
        st.markdown(
            f'<div class="footnote">{n:,} held-out images. <b>NPV is the number '
            "that matters</b>: this is a rule-out aid. A negative is trustworthy; "
            f"a positive needs a radiologist, since about {1 - t['precision_ppv']:.0%} "
            "of flagged films are false alarms. Accuracy is not shown because "
            f"{1 - (cm['tp'] + cm['fn']) / n:.0%} of the set is negative — always "
            "answering &ldquo;no pneumonia&rdquo; would score that well.</div>",
            unsafe_allow_html=True)
    st.stop()

# ---- result state ----------------------------------------------------------
try:
    raw = read_upload(upload)
except Exception as exc:  # noqa: BLE001 - surface any decode problem to the user
    st.error(f"Could not read that file: {exc}")
    st.stop()

with st.spinner("Analysing …"):
    r = analyse(model, explainers[method], raw, size, clip, grid, threshold)

method_name = "Grad-CAM" if method == "gradcam" else "Grad-CAM++"
st.markdown(theme.eyebrow("Result"), unsafe_allow_html=True)
st.markdown(theme.verdict_card(r["label"], r["prob"], threshold, r["confidence"]),
            unsafe_allow_html=True)

st.markdown(theme.panels([
    ("Uploaded", f"{raw.shape[1]}×{raw.shape[0]} → {size}×{size}", r["original"]),
    ("After CLAHE", f"clip {clip}, grid {grid}", r["clahe"]),
    (method_name, f"evidence for {r['label'].title()}", r["overlay"]),
]), unsafe_allow_html=True)

st.markdown(theme.readout([
    ("P(pneumonia)", f"{r['prob']:.3f}", "raw model output"),
    ("Threshold", f"{threshold:.3f}", "tuned on validation"),
    ("Confidence", f"{r['confidence']:.1%}", "in the call above"),
]), unsafe_allow_html=True)

with st.expander("How to read this"):
    sens = f"{metrics['test']['sensitivity_recall']:.2f}" if metrics else "—"
    spec = f"{metrics['test']['specificity']:.2f}" if metrics else "—"
    st.markdown(f"""
**The heatmap shows evidence for the predicted class**, not for pneumonia in
general. Red marks the regions that pushed the score toward **{r['label']}**. The
gradient is taken on the pre-sigmoid logit, so a confident prediction still
produces a usable map instead of a washed-out one.

**The probability is not a diagnosis.** At the tuned threshold this model reaches
sensitivity {sens} and specificity {spec} on held-out data. Roughly half of
flagged films are false alarms, so a positive means *a radiologist should look*,
not *this patient has pneumonia*.

**Resolution limits the heatmap.** DenseNet121 downsamples by 32, so the map is
{size // 32}×{size // 32} cells upsampled to {size}×{size}. It localises a region,
never a precise boundary.

**Out-of-distribution inputs give meaningless answers.** The model was trained on
frontal chest radiographs of a predominantly adult population. A lateral view, a
paediatric film, or anything that is not a chest X-ray will still return a
confident-looking number.
""")

st.markdown(
    f'<div class="footnote">Model <code>{choice.stem}</code> &middot; input '
    f"{size}×{size} &middot; CLAHE clip {clip}, grid {grid} &middot; threshold "
    f"{threshold:.3f} &middot; {method_name}</div>", unsafe_allow_html=True)
