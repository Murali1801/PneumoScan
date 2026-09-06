"""Presentation layer for the PneumoScan demo.

Kept separate from `streamlit_app.py` so the inference path stays readable and
testable: nothing in here touches the model.

Light, cool-neutral palette. Radiographs are the darkest thing on screen, so the
interface stays pale and lets the imagery carry the attention. Colour is spent
only where it means something: teal for the product, amber for caution, red and
green for the two verdicts. Depth comes from hairline borders and a barely-there
shadow rather than heavy panels.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
LINE = "#E4E9F0"
TEAL = "#0D9488"
RED = "#DC2626"
GREEN = "#059669"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #F7F9FB; }

/* Streamlit's default chrome fights a full-bleed layout */
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding: 2.2rem 3rem 4rem; max-width: 1180px; }

/* ---------- hero ---------- */
.hero { display: flex; align-items: center; gap: .95rem; margin-bottom: .35rem; }
.hero-mark {
  width: 42px; height: 42px; border-radius: 12px; flex: none;
  background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
  display: grid; place-items: center; font-size: 21px;
  box-shadow: 0 4px 12px rgba(13,148,136,.24);
}
.hero-title { font-size: 1.85rem; font-weight: 700; letter-spacing: -.022em;
              line-height: 1.1; color: #0F172A; }
.hero-sub { color: #64748B; font-size: .84rem; margin: .1rem 0 0; letter-spacing: .01em; }

.notice {
  display: flex; gap: .6rem; align-items: flex-start;
  border: 1px solid #FDE3A7; background: #FFFBEB;
  border-radius: 11px; padding: .7rem .9rem; margin: 1.3rem 0 1.9rem;
  font-size: .8rem; color: #8A5A08; line-height: 1.5;
}
.notice b { color: #B45309; font-weight: 600; }

/* ---------- section labels ---------- */
.eyebrow {
  font-size: .69rem; font-weight: 600; letter-spacing: .13em;
  text-transform: uppercase; color: #94A3B8; margin: 2.1rem 0 .85rem;
}

/* ---------- verdict ---------- */
.verdict {
  border-radius: 16px; padding: 1.5rem 1.7rem; margin: .4rem 0 1.5rem;
  border: 1px solid var(--edge); background: var(--wash);
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.verdict-top { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
.verdict-label {
  display: flex; align-items: center; gap: .6rem;
  font-size: 1.4rem; font-weight: 700; letter-spacing: -.02em; color: var(--tone);
}
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--tone);
       box-shadow: 0 0 0 4px var(--halo); flex: none; }
.verdict-pct { font-size: 2.5rem; font-weight: 700; letter-spacing: -.035em;
               color: var(--tone); line-height: 1; font-variant-numeric: tabular-nums; }
.verdict-note { color: #475569; font-size: .82rem; margin-top: .55rem; line-height: 1.5; }

/* probability track with the operating point marked */
.track { position: relative; height: 7px; border-radius: 99px;
         background: #E4E9F0; margin: 1.5rem 0 .5rem; overflow: visible; }
.track-fill { position: absolute; inset: 0 auto 0 0; border-radius: 99px; background: var(--tone); }
.track-mark { position: absolute; top: -6px; width: 2px; height: 19px;
              background: #0F172A; border-radius: 2px; opacity: .8; }
.track-scale { display: flex; justify-content: space-between;
               font-size: .68rem; color: #94A3B8; margin-top: .45rem;
               font-variant-numeric: tabular-nums; }

/* ---------- image panels ---------- */
.panels { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.panel { border: 1px solid #E4E9F0; border-radius: 14px; overflow: hidden;
         background: #FFFFFF; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.panel img { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }
.panel-cap { padding: .62rem .8rem .68rem; border-top: 1px solid #EEF1F6; }
.panel-cap b { display: block; font-size: .78rem; font-weight: 600; color: #0F172A; }
.panel-cap span { font-size: .69rem; color: #94A3B8; }

/* ---------- stat cards ---------- */
.stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: .7rem; }
.stat { border: 1px solid #E4E9F0; border-radius: 12px; padding: .85rem .9rem;
        background: #FFFFFF; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.stat-k { font-size: .68rem; font-weight: 600; letter-spacing: .07em;
          text-transform: uppercase; color: #94A3B8; }
.stat-v { font-size: 1.5rem; font-weight: 700; letter-spacing: -.03em; color: #0F172A;
          margin: .28rem 0 .12rem; font-variant-numeric: tabular-nums; }
.stat-d { font-size: .7rem; color: #64748B; line-height: 1.35; }
.stat.lead { border-color: #99F6E4; background: #F0FDFA; }
.stat.lead .stat-v { color: #0D9488; }

.readout { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin-top: 1rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E4E9F0; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.side-h { font-size: .68rem; font-weight: 600; letter-spacing: .13em;
          text-transform: uppercase; color: #94A3B8; margin: 1.5rem 0 .5rem; }
.side-h:first-child { margin-top: 0; }
.side-note { font-size: .71rem; color: #94A3B8; line-height: 1.45; margin-top: .35rem; }
.side-note b { color: #64748B; font-weight: 600; }

/* ---------- controls ---------- */
[data-testid="stFileUploader"] section {
  border: 1.5px dashed #CBD5E1; border-radius: 14px; background: #FFFFFF;
  padding: 1.4rem; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: #0D9488; background: #F0FDFA; }
[data-testid="stFileUploader"] label { font-size: .8rem; }

div[data-testid="stExpander"] details {
  border: 1px solid #E4E9F0; border-radius: 12px; background: #FFFFFF;
}
div[data-testid="stExpander"] summary { font-size: .8rem; font-weight: 500; }

.stAlert { border-radius: 11px; font-size: .82rem; }

.footnote { color: #94A3B8; font-size: .71rem; margin-top: 2.2rem;
            border-top: 1px solid #E4E9F0; padding-top: .9rem; line-height: 1.6; }
.footnote b { color: #64748B; font-weight: 600; }
.footnote code { color: #475569; background: #F1F5F9; padding: .07rem .34rem;
                 border-radius: 4px; font-size: .95em; }

@media (max-width: 900px) {
  .block-container { padding: 1.4rem 1.1rem 3rem; }
  .panels, .stats, .readout { grid-template-columns: 1fr; }
  .verdict-pct { font-size: 2rem; }
}
</style>
"""


def to_data_uri(arr: np.ndarray) -> str:
    """uint8 array -> inline PNG. RGB is converted to BGR for cv2's encoder."""
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def hero() -> str:
    return """
<div class="hero">
  <div class="hero-mark">🫁</div>
  <div>
    <div class="hero-title">PneumoScan</div>
    <p class="hero-sub">Explainable pneumonia screening &middot; DenseNet121 + CLAHE + Grad-CAM
       &middot; RSNA Pneumonia Detection Challenge</p>
  </div>
</div>
<div class="notice">
  <span>&#9888;</span>
  <div><b>Research demonstration only — not a medical device.</b> A student project
  trained on a single public dataset. It must not be used to make or support any
  clinical decision.</div>
</div>
"""


def verdict_card(label: str, prob: float, threshold: float, confidence: float) -> str:
    """The headline result, with the probability drawn against its operating point.

    Showing where the score sits relative to the threshold - rather than a bare
    percentage - is the difference between a number and a decision the viewer can
    actually interrogate.
    """
    positive = label == "PNEUMONIA"
    tone = RED if positive else GREEN
    halo = "rgba(220,38,38,.14)" if positive else "rgba(5,150,105,.14)"
    wash = "#FEF2F2" if positive else "#ECFDF5"
    edge = "#FECACA" if positive else "#A7F3D0"
    note = ("Above the operating point — a radiologist should review this film."
            if positive else
            "Below the operating point — no lung opacity detected.")
    return f"""
<div class="verdict" style="--tone:{tone};--halo:{halo};--wash:{wash};--edge:{edge}">
  <div class="verdict-top">
    <div>
      <div class="verdict-label"><span class="dot"></span>{label}</div>
      <div class="verdict-note">{note}<br>Confidence in this call: {confidence:.1%}</div>
    </div>
    <div class="verdict-pct">{prob:.1%}</div>
  </div>
  <div class="track">
    <div class="track-fill" style="width:{prob * 100:.2f}%"></div>
    <div class="track-mark" style="left:{threshold * 100:.2f}%"></div>
  </div>
  <div class="track-scale">
    <span>0.0</span><span>threshold {threshold:.3f}</span><span>1.0</span>
  </div>
</div>
"""


def panels(items: list[tuple[str, str, np.ndarray]]) -> str:
    cells = "".join(
        f'<div class="panel"><img src="{to_data_uri(arr)}" alt="{title}">'
        f'<div class="panel-cap"><b>{title}</b><span>{sub}</span></div></div>'
        for title, sub, arr in items)
    return f'<div class="panels">{cells}</div>'


def stats(rows: list[tuple[str, str, str]], lead: str | None = None) -> str:
    cells = "".join(
        f'<div class="stat{" lead" if k == lead else ""}">'
        f'<div class="stat-k">{k}</div><div class="stat-v">{v}</div>'
        f'<div class="stat-d">{d}</div></div>'
        for k, v, d in rows)
    return f'<div class="stats">{cells}</div>'


def readout(rows: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="stat-k">{k}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-d">{d}</div></div>'
        for k, v, d in rows)
    return f'<div class="readout">{cells}</div>'


def eyebrow(text: str) -> str:
    return f'<div class="eyebrow">{text}</div>'


def side_header(text: str) -> str:
    return f'<div class="side-h">{text}</div>'
