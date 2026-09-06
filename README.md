# Chest X-ray Screening with Grad-CAM Explainability

Binary pneumonia screening on the **Kermany** paediatric chest X-ray dataset, with
CLAHE preprocessing, DenseNet121 / EfficientNet-B0 transfer learning, **Grad-CAM**
visual explanations, and a TensorFlow Lite export for the mobile app.

*Data Science in Healthcare mini project — Department of Computer Engineering,
St. John College of Engineering and Management.
Murlidhar Acharya (A-02) · Urvi Khandelwal (A-46) · Srushti Raut (B-35).*

---

## Pipeline

```
Kaggle: Kermany chest X-rays
        │
        ├─ 1. prepare_data.py ── CLAHE ──► 224×224 PNG
        │                    └─ patient-grouped, class-stratified splits
        │
        ├─ 2. train.py ──► phase 1 frozen head → phase 2 fine-tune
        │                  DenseNet121 | EfficientNet-B0
        │                  → checkpoints/<run>.keras
        │                  → reports/<run>/  metrics.json, ROC/PR, confusion matrix
        │
        ├─ 3. make_gradcam_figures.py ──► Grad-CAM / Grad-CAM++ panels
        │
        └─ 4. export_tflite.py ──► <run>.tflite  +  <run>_dynamic.tflite
```

## Where it runs

Training needs TensorFlow, and TensorFlow has **no wheel for Python 3.13/3.14**
and no native Windows GPU support since 2.11. So:

* **Training → Google Colab (free T4).** Open `notebooks/train_colab.ipynb`,
  set the runtime to T4 GPU, and run the cells top to bottom. Full run
  (download + both backbones + Grad-CAM + TFLite) is roughly 45–60 minutes.
* **Locally** you can still run everything *except* training, in a Python 3.11/3.12
  environment:

  ```bash
  py -3.11 -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

## Running it by hand

```bash
python scripts/prepare_data.py --download          # needs ~/.kaggle/kaggle.json
python src/train.py --backbone densenet121
python src/train.py --backbone efficientnetb0
python scripts/make_gradcam_figures.py --backbone densenet121
python src/export_tflite.py --backbone densenet121
```

Every hyperparameter in `src/config.py` is exposed as a CLI flag, e.g.
`--clahe-clip 3.0 --batch-size 16 --finetune-epochs 30 --no-use-class-weights`.

## Design decisions worth defending in the viva

**Patient-grouped splits.** Kermany ships several images per child
(`person23_bacteria_76.jpeg`, `person23_bacteria_77.jpeg`). A random 90/10 split
puts one of a patient's images in train and another in validation, so the model
scores well by recognising the *patient* rather than the disease. `src/data.py`
extracts a patient id from every filename and splits with
`StratifiedGroupKFold`, then asserts that no patient id appears in two splits.

**The official validation folder is unusable.** It contains 16 images — 8 per
class. A threshold or an early-stopping decision made on 16 images is noise, so
it is merged back into train and a real 10% validation set is carved out.

**The test folder is never touched.** Thresholds are chosen on validation
(Youden's J), model selection is on validation AUROC. The test set is only ever
reported on, with a bootstrap 95% CI on AUROC.

**CLAHE, not global histogram equalisation.** Global equalisation blows out the
mediastinum and flattens the lung fields. CLAHE equalises inside 8×8 tiles and
clips the histogram first, so consolidation and infiltrates become visible
without amplifying noise. The exact same `clahe_image()` runs offline for
training and at inference in the demo — one implementation, so there is no
train/serve skew.

**No horizontal flip.** Mirroring a chest radiograph produces anatomically
impossible images (heart on the right) and teaches the model to discard
laterality. Rotation ±11°, translation 6%, zoom 10% and contrast jitter only.

**BatchNorm stays frozen during fine-tuning.** With batch size 32 on ~4700
images, recomputing BatchNorm statistics destabilises the pretrained features;
the backbone is called with `training=False` throughout.

**Grad-CAM differentiates the logit, not the sigmoid.** A confident sigmoid sits
at ~1.0 where its gradient is ~0, which washes the heatmap out. `src/gradcam.py`
takes the gradient of the pre-sigmoid score, and flips its sign for a NORMAL
prediction so the map shows evidence for whichever class was predicted.

**Threshold at 0.5 is reported too.** `metrics.json` carries both `test` (tuned
threshold) and `test_at_0.5`, so the effect of the operating point is visible
rather than hidden.

## Layout

```
src/config.py        all hyperparameters, mirrored as CLI flags
src/preprocess.py    CLAHE — shared by training, figures and inference
src/data.py          indexing + patient-grouped stratified splitting (no TF)
src/pipeline.py      tf.data input pipeline and augmentation
src/model.py         DenseNet121 / EfficientNet-B0 heads; named feature_map + logits
src/train.py         two-phase training
src/evaluate.py      threshold selection, metrics, ROC/PR/confusion plots
src/gradcam.py       Grad-CAM and Grad-CAM++ + figure rendering
src/export_tflite.py SavedModel → TFLite, with Keras-parity check
scripts/prepare_data.py        download + CLAHE + splits
scripts/make_gradcam_figures.py explanation panels
notebooks/train_colab.ipynb    the notebook that actually trains
app/                           Streamlit demo (stage 5, not built yet)
```

## Model input contract

Everything downstream depends on this: the model takes **float32 RGB in [0, 255],
224×224, CLAHE already applied**. Each backbone's own normalisation is baked in
as the first layers of the graph, so the tf.data pipeline, the Grad-CAM code and
the `.tflite` file all feed the network identical tensors.

## Outputs

| File | What it is |
|---|---|
| `reports/dataset_summary.csv` | images and *patients* per split and class |
| `reports/clahe_examples.png` | original / CLAHE / added-contrast figure |
| `reports/<run>/metrics.json` | full metric report, both thresholds, AUROC CI |
| `reports/<run>/training_curves.png` | loss, AUC, accuracy across both phases |
| `reports/<run>/test_curves.png` | ROC and precision-recall |
| `reports/<run>/test_confusion.png` | confusion matrix + score distribution |
| `reports/<run>/gradcam_*.png` | explanation panels, including the failure cases |
| `reports/backbone_comparison.csv` | DenseNet121 vs EfficientNet-B0 |
| `checkpoints/<run>.tflite` | mobile model (float32 and int8-weights variants) |

## Not built yet

* `app/` — the Streamlit upload → prediction → confidence → heatmap demo
* the Android/Flutter wrapper around the `.tflite` model
