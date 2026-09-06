# PneumoScan — Explainable Chest X-ray Pneumonia Screening

Binary pneumonia screening on **adult** chest radiographs from the RSNA Pneumonia
Detection Challenge, with CLAHE preprocessing, DenseNet121 transfer learning,
**Grad-CAM scored against radiologist bounding boxes**, and a TensorFlow Lite
export for the mobile app.

*Data Science in Healthcare mini project — Department of Computer Engineering,
St. John College of Engineering and Management.
Murlidhar Acharya (A-02) · Urvi Khandelwal (A-46) · Srushti Raut (B-35).*

---

## Pipeline

```
Kaggle: RSNA Pneumonia Detection Challenge (26,684 adult DICOMs)
        │
        ├─ 1. prepare_data.py ── DICOM → CLAHE ──► 224×224 PNG
        │                    ├─ 70/15/15 stratified, patient-grouped splits
        │                    ├─ radiologist boxes rescaled into the resized frame
        │                    └─ cohort table from DICOM headers (age/sex/view)
        │
        ├─ 2. train.py ──► phase 1 frozen head → phase 2 fine-tune
        │                  → checkpoints/<run>.keras
        │                  → reports/<run>/  metrics.json, ROC/PR, confusion matrix
        │
        ├─ 3. make_gradcam_figures.py ──► Grad-CAM / Grad-CAM++ panels
        │                             └─► localization.json — the XAI *metric*
        │
        └─ 4. export_tflite.py ──► <run>.tflite  +  <run>_dynamic.tflite
```

## The dataset

| | |
|---|---|
| Source | RSNA Pneumonia Detection Challenge (Kaggle competition) |
| Origin | Re-annotated subset of NIH ChestX-ray, boxes drawn by radiologists |
| Images | 26,684 labelled DICOMs, 1024×1024 |
| Population | **Adult** — verify it yourself in `reports/cohort.csv` |
| Labels | `Target` 1 = Lung Opacity (~6.0k), 0 = Normal (~8.9k) + No Lung Opacity/Not Normal (~11.8k) |
| Boxes | ~9,500, on the positive cases only |

The competition's own `stage_2_test_images/` is unlabelled — it was the
leaderboard set — so all three splits are made here from the 26,684 labelled
images.

**Accept the competition rules once** before downloading, or Kaggle returns 403:
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules

## Where it runs

Training needs TensorFlow, and TensorFlow has **no wheel for Python 3.13/3.14**
and no native Windows GPU support since 2.11. So:

* **Training → Google Colab (free T4).** Open `notebooks/train_colab.ipynb`, set
  the runtime to T4 GPU, run top to bottom. Budget **90–120 minutes**; section 5
  offers a 20-minute rehearsal on a subsample first.
* **Locally** everything except training works in Python 3.11/3.12:

  ```bash
  py -3.11 -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

## Verify before you spend an hour

```bash
python scripts/selftest.py
```

Generates a small synthetic RSNA-shaped dataset — real DICOMs, real competition
CSVs — and runs all four stages against it in about two minutes, checking each
one produced its outputs. `SELF-TEST PASSED` means the code and the environment
work; anything that fails afterwards is data or credentials, not the pipeline.
The synthetic DICOMs are 512×512 rather than 1024 on purpose, so box rescaling
has to actually work instead of passing on a 1:1 scale factor.

## Running it by hand

```bash
python scripts/prepare_data.py --download           # needs Kaggle creds + rules accepted
python src/train.py --subsample-train 3000 --tag rehearsal   # 20-min dry run
python src/train.py
python scripts/make_gradcam_figures.py
python src/export_tflite.py
```

Every field in `src/config.py` is a CLI flag, e.g. `--clahe-clip 3.0
--batch-size 16 --iou-threshold 0.3 --exclude-not-normal --no-use-class-weights`.

## Design decisions worth defending in the viva

**Why DenseNet121, and only DenseNet121.** CheXNet (Rajpurkar et al., 2017)
established this architecture for pneumonia detection on NIH ChestX-ray14, and
RSNA is a re-annotated subset of exactly that collection — so this is the
reference architecture evaluated on data from the collection it was validated on.
Khadidos et al. (2026) report EfficientNet-B0 as more deployment-efficient, but
on paediatric data; a comparable evaluation on adult radiographs is left as
future work.

**Grad-CAM is measured, not admired.** RSNA's boxes make explainability
quantitative. `src/localization.py` reports three metrics — pointing game
(is the hottest pixel inside a box?), energy pointing game (what share of the
heatmap's mass is inside?), and IoU@τ — each against a **chance baseline** equal
to the boxes' own share of the image. Boxes cover roughly a quarter of a film, so
a random heatmap already "scores" ~0.25 on the pointing game; the number that
means something is the **lift** over that baseline. A metric reported without its
baseline is not evidence, and this is the single strongest thing in the project.

**"Not Normal" films are kept as negatives.** A third of the dataset is abnormal
for some reason other than pneumonia. Dropping them makes the task pneumonia-vs-
healthy, which is easier and clinically dishonest — a screening tool has to tell
pneumonia from *other pathology*, not just from healthy lungs. Run with
`--exclude-not-normal` to measure exactly how much of your score came from that
easier problem.

**The cohort table is evidence, not an assertion.** `reports/cohort.csv` reads
age, sex and view straight from the DICOM headers, so "this is an adult dataset"
is something the report demonstrates rather than claims.

**MONOCHROME1 is handled.** Some DICOMs store an inverted greyscale ramp. Ignore
the photometric interpretation and those X-rays silently arrive as photographic
negatives — a class of bug that quietly costs accuracy and is nearly invisible in
a thumbnail.

**Patient-grouped, stratified splits, asserted in code.** `StratifiedGroupKFold`
on `patientId`, then an assertion that no group spans two splits. RSNA issues one
patientId per image so this is mostly a safety net — but the guarantee is
enforced rather than assumed. *Honest caveat:* RSNA is re-annotated from NIH,
where one person can contribute several studies, and that mapping is not
published, so a little same-person leakage is undetectable from this data alone.

**Byte-level duplicate check.** Ids can't detect the same radiograph filed twice
under different names, so `prepare_data.py` md5-hashes every file and reports any
image appearing in two splits.

**The test set is never touched.** Thresholds come from validation (Youden's J),
model selection from validation AUROC. Test is reported once, with a bootstrap
95% CI, and `metrics.json` also carries the numbers at a plain 0.5 threshold so
the effect of the operating point is visible rather than hidden.

**CLAHE, not global histogram equalisation.** Global equalisation blows out the
mediastinum and flattens the lung fields. CLAHE equalises inside 8×8 tiles and
clips the histogram first. The same `clahe_image()` runs offline for training and
at inference in the demo — one implementation, no train/serve skew.

**No horizontal flip.** Mirroring a chest radiograph produces anatomically
impossible images and teaches the model to discard laterality. Rotation ±11°,
translation 6%, zoom 10%, contrast jitter only.

**BatchNorm frozen during fine-tuning.** With batch size 32, recomputing
BatchNorm statistics destabilises pretrained features; the backbone is called
with `training=False` throughout.

**Grad-CAM differentiates the logit, not the sigmoid.** A confident sigmoid sits
at ~1.0 where its gradient is ~0, which washes the heatmap out. The gradient is
taken on the pre-sigmoid score, with the sign flipped for a NORMAL prediction so
the map shows evidence for whichever class was predicted.

## Layout

```
src/config.py         all hyperparameters, mirrored as CLI flags
src/preprocess.py     DICOM reading + CLAHE — shared by training, figures, inference
src/data.py           RSNA indexing, boxes, splits, duplicate check (no TF)
src/pipeline.py       tf.data input pipeline and augmentation
src/model.py          DenseNet121; named feature_map + logits layers
src/train.py          two-phase training
src/evaluate.py       threshold selection, metrics, ROC/PR/confusion plots
src/gradcam.py        Grad-CAM and Grad-CAM++ + figure rendering
src/localization.py   pointing game / energy / IoU against the radiologist boxes
src/export_tflite.py  SavedModel → TFLite, with Keras-parity check
scripts/selftest.py              end-to-end check on synthetic data
scripts/prepare_data.py          download + DICOM + CLAHE + splits + boxes
scripts/make_gradcam_figures.py  explanation panels + the localisation report
notebooks/train_colab.ipynb      the notebook that actually trains
app/                             Streamlit demo (stage 5, not built yet)
```

## Model input contract

Everything downstream depends on this: the model takes **float32 RGB in [0, 255],
224×224, CLAHE already applied**. Each backbone's own normalisation is baked in
as the first layers of the graph, so the tf.data pipeline, the Grad-CAM code and
the `.tflite` file all feed the network identical tensors.

## Outputs

| File | What it is |
|---|---|
| `reports/dataset_summary.csv` | images and boxes per split and class |
| `reports/detailed_class_summary.csv` | the three-way RSNA class breakdown |
| `reports/cohort.csv` | age / sex / view — the proof the data is adult |
| `reports/clahe_examples.png` | original / CLAHE / radiologist boxes |
| `reports/<run>/metrics.json` | full metric report, both thresholds, AUROC CI |
| `reports/<run>/localization.csv` | **the XAI metric**, Grad-CAM vs Grad-CAM++ |
| `reports/<run>/training_curves.png` | loss, AUC, accuracy across both phases |
| `reports/<run>/test_curves.png` | ROC and precision-recall |
| `reports/<run>/gradcam_*.png` | explanation panels with boxes drawn |
| `checkpoints/<run>.tflite` | mobile model (float32 and int8-weights variants) |

## Limitations to state in the report

* Single dataset, one annotation effort — no external validation set.
* "Pneumonia" here means *radiographic lung opacity*, which is a radiological
  finding, not a confirmed clinical diagnosis.
* NIH-derived, so a small amount of same-patient leakage across splits cannot be
  ruled out.
* Screening aid only. Not a diagnostic device.

## Not built yet

* `app/` — the Streamlit upload → prediction → confidence → heatmap demo
* the Android/Flutter wrapper around the `.tflite` model
