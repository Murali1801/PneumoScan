# PneumoScan — Explainable Chest X-ray Pneumonia Screening

Binary pneumonia screening on **predominantly adult** chest radiographs from the
RSNA Pneumonia Detection Challenge, with CLAHE preprocessing, DenseNet121 transfer learning,
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
| Population | **Predominantly adult** — median 49, IQR 35–60, but 4.4% under 18 (`reports/cohort.csv`) |
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
python scripts/cohort.py                            # age/sex/view table
python src/train.py --subsample-train 3000 --tag rehearsal   # 20-min dry run
python src/train.py
python scripts/make_gradcam_figures.py
python src/export_tflite.py
```

Two flags matter for long GPU runs:

* `--mixed-precision` — float16 compute on tensor cores, roughly 1.5x faster on a
  T4 and half the activation memory. Master weights stay float32, and the saved
  checkpoint is rebuilt as a plain float32 graph so Grad-CAM and the TFLite
  converter never see float16.
* `--resume` — continue an interrupted run. The epoch to restart from is read out
  of the phase's CSV log, and phase 1 is checkpointed separately so a failure
  during fine-tuning never costs the warm-up. Point `--ckpt-dir` and `--out-dir`
  at Google Drive and a dropped Colab session costs one epoch.

Every field in `src/config.py` is a CLI flag, e.g. `--clahe-clip 3.0
--batch-size 16 --iou-threshold 0.3 --exclude-not-normal --no-use-class-weights`.

## Results

Two runs, identical except for input resolution. Test set: **4,447 images,
1,008 positive (22.7% prevalence)**.

| Metric | 224px | 320px | change |
|---|---|---|---|
| **AUROC** | 0.8765 | **0.8791** | +0.0026 |
| AUROC 95% CI | 0.865 – 0.888 | 0.867 – 0.890 | overlapping |
| AUPRC | 0.7056 | 0.7093 | +0.0037 |
| Sensitivity | 0.8214 | 0.8264 | +0.0050 |
| Specificity | 0.7738 | 0.7671 | −0.0067 |
| NPV | 0.9366 | 0.9378 | +0.0011 |
| PPV | 0.5156 | 0.5098 | −0.0058 |
| Accuracy | 0.7846 | 0.7805 | −0.0040 |
| Training time | 55 min | 163 min | 3× |

### Grad-CAM localisation vs radiologist boxes

| Metric | 224px | 320px | change |
|---|---|---|---|
| **Pointing game** | 0.435 | **0.712** | **+0.277** |
| **Lift over chance** | 3.3× | **5.4×** | +2.1× |
| Energy pointing game | 0.304 | 0.447 | +0.143 |
| IoU@0.5 | 0.267 | 0.321 | +0.054 |
| Chance baseline (box coverage) | 0.131 | 0.131 | unchanged |

### The finding

**Raising input resolution left classification unchanged but transformed
explanation quality.** The 320px AUROC (0.8791) sits inside the 224px confidence
interval (0.865–0.888) and the two intervals overlap almost entirely, so no
significant classification improvement can be claimed. Over the same images and
against an unchanged chance baseline, Grad-CAM's pointing game rose from 0.435 to
0.712 — from 3.3× to 5.4× chance.

The interpretation: the classifier's discriminative ceiling is set by **label
consistency** — radiologists disagree about what counts as lung opacity — while
the explanation's fidelity was limited by **feature-map resolution**. DenseNet121
downsamples by a fixed factor of 32, so 224px input yields a 7×7 map and 320px a
10×10 one: twice as many cells over the same anatomy.

This is the project's main result, and it argues for reporting explainability
quality as a metric in its own right rather than as a by-product of accuracy.

**Grad-CAM++ did not beat Grad-CAM** at either resolution (0.712 vs 0.713 at
320px) — reported as measured rather than quietly dropping the loser.

### Which model to use

**320px**, because the explainability metric is this project's contribution and
it nearly doubled, at no cost to classification. The 224px model remains the
better choice if training or inference cost matters — it is 3× faster to train
for statistically identical classification.

### Reporting note

Do not lead with accuracy. 77.3% of the test set is negative, so predicting "no
pneumonia" for every image scores 0.773 — both models' ~0.78 is barely a point
above a classifier that does nothing. AUROC and the sensitivity/specificity pair
are the honest headline, and **NPV 0.938** is the framing that fits the tool:
when it says no pneumonia it is right 94% of the time, which is a rule-out
screening aid. PPV 0.510 means roughly half of flagged films are false alarms, so
it cannot stand alone as a diagnostic.

### Cohort — read this before claiming "adult"

Measured from 3,000 DICOM headers (`reports/cohort.csv`):

| | |
|---|---|
| Age median (IQR) | **49 (35 – 60)** |
| Under 18 | **4.4%** |
| Reported range | 2 – 155 |
| Male | 57.5% |
| PA / AP view | 55% / 45% |

The cohort is **predominantly, not exclusively, adult** — 4.4% are paediatric, so
"adult chest radiographs" overstates it; "predominantly adult (median 49, IQR
35–60)" is accurate. The reported maximum of 155 years is impossible: age fields
in the NIH ChestX-ray collection RSNA is drawn from are known to contain
data-entry errors. Both facts belong in the limitations section.

### Training

16 epochs (3 frozen-head + 13 fine-tuning) at 320px, best validation AUROC 0.8892
at epoch 12, drifting down afterwards — converged, so more epochs would not help.
The 224px run behaved the same way, peaking at epoch 12 of 15.

Mixed precision was less effective than expected: 163 minutes against a predicted
~75. DenseNet's dense concatenations make poor use of tensor cores, and
checkpointing to Drive each epoch adds overhead.

### TFLite

| File | Size | Max abs. difference vs Keras |
|---|---|---|
| `densenet121_320.tflite` | 27.9 MB | 1.0e-06 |
| `densenet121_320_dynamic.tflite` | 7.4 MB | 2.9e-02 |

Identical sizes at both resolutions — input size does not change the parameter
count. The 320px model expects 320×320 input; the demo app must match.

## The demo app

```bash
streamlit run app/streamlit_app.py
```

Upload a chest X-ray (DICOM, PNG or JPEG) and get a prediction, a confidence, and
a Grad-CAM heatmap showing what drove it.

It reads everything from the checkpoint and its run directory rather than
hard-coding anything:

* **Input size** comes from `model.input_shape`, so a 224px and a 320px
  checkpoint both work unchanged. Hard-coding it is how a demo ends up silently
  feeding the network upsampled images.
* **The decision threshold** comes from the run's `metrics.json` — the value
  tuned on validation. Falling back to 0.5 would quietly move the operating point
  that the reported sensitivity and specificity belong to.
* **CLAHE** uses the same `clahe_image()` the training set was built with.
* The checkpoint picker is **ordered by held-out AUROC**, and `_phase1` and
  `_rehearsal` checkpoints are hidden so the demo cannot open on a warm-up or a
  deliberately under-trained model.

It looks for models in `checkpoints/` and `results/checkpoints/`, and their
metrics in `reports/` and `results/reports/`. Both are gitignored, so copy the
trained checkpoint and its report folder in after downloading them from Colab.

Before the upload it shows the model's held-out AUROC, sensitivity, specificity,
NPV and PPV, so the number it gives you is interpretable rather than bare. A
prominent notice states it is a research demonstration and not a medical device.

## Hosting it online

Two separate things: the **code** and the **live demo**.

### Code → GitHub

The repo is already a git repo. `checkpoints/`, `reports/*/`, `data/` and
`build/` are gitignored, so a push carries the pipeline and not the 30 MB
checkpoints or the 3.7 GB dataset.

### Demo → Hugging Face Spaces

```bash
python scripts/prepare_space.py
```

Builds `build/space/` — a self-contained ~30 MB folder holding only what the demo
needs, then prints the push commands. Create a Space at
[huggingface.co/new-space](https://huggingface.co/new-space) with SDK
**Streamlit**, then from `build/space/`:

```bash
git init && git lfs install && git lfs track "*.keras"
git add -A && git commit -m "PneumoScan demo"
git remote add origin https://huggingface.co/spaces/<you>/pneumoscan
git push -u origin main
```

**Why Spaces rather than Streamlit Community Cloud.** Grad-CAM needs gradients,
so the demo has to run full TensorFlow — the 7 MB TFLite model can infer but
cannot backpropagate. TensorFlow plus DenseNet121 wants roughly 700 MB of RAM,
which is uncomfortably close to Streamlit Cloud's ~1 GB ceiling. A free Spaces
CPU box has 16 GB. Streamlit Cloud will probably work; Spaces will.

**Why a separate build rather than deploying the repo.** The app imports five
packages; the training pipeline also needs pandas, scikit-learn, matplotlib and
kaggle. Deploying the repo's `requirements.txt` would install roughly a gigabyte
the demo never touches. The generated `requirements.txt` also pins
**`tensorflow-cpu`** — on Linux the default `tensorflow` wheel drags in the CUDA
runtime, hundreds of megabytes of GPU libraries a CPU host cannot use.

`prepare_space.py` parses the copied files and fails the build if any project
module they import is missing, so an incomplete build is caught locally rather
than after a push.

Notes: the 30 MB checkpoint needs **Git LFS** (Spaces expects it above 10 MB);
free Spaces sleep after inactivity and take ~30 s to wake; first load takes
~20 s while TensorFlow imports.

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
to the boxes' own share of the image. Measured on this dataset the boxes cover
**13.1%** of a film, so a random heatmap already "scores" 0.131 on the pointing
game; the number that means something is the **lift** over that baseline. A metric reported without its
baseline is not evidence, and this is the single strongest thing in the project.

**"Not Normal" films are kept as negatives.** A third of the dataset is abnormal
for some reason other than pneumonia. Dropping them makes the task pneumonia-vs-
healthy, which is easier and clinically dishonest — a screening tool has to tell
pneumonia from *other pathology*, not just from healthy lungs. Run with
`--exclude-not-normal` to measure exactly how much of your score came from that
easier problem.

**The cohort table is evidence, not an assertion.** `reports/cohort.csv` reads
age, sex and view straight from the DICOM headers - and measuring rather than
assuming is what revealed that 4.4% of the cohort is under 18 and that some ages
are impossible (a reported maximum of 155). "Adult dataset" was the assumption;
"predominantly adult" is what the data supports.

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
app/streamlit_app.py             upload → prediction → confidence → heatmap demo
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

* **Predominantly, not exclusively, adult**: 4.4% of the cohort is under 18, and
  some reported ages are impossible (maximum 155), a known data-quality issue in
  the NIH ChestX-ray collection RSNA is drawn from.
* Single dataset, one annotation effort — no external validation set.
* "Pneumonia" here means *radiographic lung opacity*, which is a radiological
  finding, not a confirmed clinical diagnosis.
* NIH-derived, so a small amount of same-patient leakage across splits cannot be
  ruled out.
* Screening aid only. Not a diagnostic device.

## Not built yet

* the Android/Flutter wrapper around the `.tflite` model
