<div align="center">

# 📦 Shipping Container Damage Detection with YOLO11

### An End-to-End Object Detection Pipeline — EDA → Training → Evaluation → Model Comparison


[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Roboflow](https://img.shields.io/badge/Source-Roboflow-6706CE?style=flat-square&logo=roboflow&logoColor=white)](https://roboflow.com)

</div>

---

## 🧭 Overview.

Shipping containers take a beating in transit — repeated handling, weather exposure, and stacking stress leave them **dented, punctured, and structurally deformed**. Manual inspection is slow, subjective, and doesn't scale.

This project trains and benchmarks **three YOLO11 variants** (`n`, `s`, `m`) to automatically **detect and localize** three damage types in container images:

| Class | Description |
|---|---|
| 🔴 **Dent** | Localized inward deformation of the container wall |
| 🔵 **Hole** | Puncture / breach in the container shell |
| 🟢 **Deframe** | Structural deformation of the container frame |

Given an image, the model answers two questions simultaneously — **what** damage is present, and **where** it is — via bounding boxes and confidence scores.

---

## 🏆 Headline Result.

> **YOLO11s wins on accuracy** (`mAP50-95 = 0.845`) while staying **~2× smaller and faster** than YOLO11m — the best accuracy-per-parameter trade-off of the three.

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Params (M) | Weights (MB) | Inference (ms/img) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **YOLO11s** | **0.972** | 0.940 | **0.958** | **0.845** | 9.43 | 18.30 | 10.97 |
| 🥈 YOLO11n | 0.932 | **0.940** | 0.956 | 0.824 | **2.59** | **5.22** | **7.35** |
| 🥉 YOLO11m | **0.972** | 0.902 | 0.952 | 0.788 | 20.06 | 38.65 | 31.55 |

*Sorted by mAP@50-95 on the held-out test set. Full breakdown in [Results](#-results--evaluation).*

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Headline Result](#-headline-result)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Training](#-model-training)
- [Results & Evaluation](#-results--evaluation)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Notebook Phases](#-notebook-phases)
- [Roadmap](#-roadmap)
- [Citation](#-citation)
- [License](#-license)

---

## 🗺️ Pipeline Architecture

```mermaid
flowchart TD
    A[📦 Raw Container Images] --> B[🔍 Dataset Integrity Checks]
    B --> C[📊 Exploratory Data Analysis]
    C --> D[🖼️ Ground-Truth Visualization]
    D --> E[🎨 Augmentation Preview]
    E --> F[⚙️ Training Configuration]

    F --> G1[🧠 Train YOLO11n]
    F --> G2[🧠 Train YOLO11s]
    F --> G3[🧠 Train YOLO11m]

    G1 --> H[✅ Validation on Test Set]
    G2 --> H
    G3 --> H

    H --> I[📈 Metrics Aggregation]
    I --> J[📊 Comparative Visualization]
    J --> K[🧩 Confusion Matrices & PR Curves]
    K --> L[🔬 Qualitative Inference Comparison]
    L --> M[⚠️ Error Analysis]
    M --> N[🏁 Final Leaderboard]
    N --> O[📤 Export Results]

    style A fill:#2C3E50,color:#fff
    style O fill:#27AE60,color:#fff
    style G1 fill:#3498DB,color:#fff
    style G2 fill:#E67E22,color:#fff
    style G3 fill:#9B59B6,color:#fff
    style N fill:#F1C40F,color:#000
```

### Model Inference Flow

```mermaid
flowchart LR
    IMG[Input Image] --> PRE[Preprocess<br/>640×640, RGB]
    PRE --> YOLO[YOLO11 Backbone<br/>+ Neck + Detection Head]
    YOLO --> NMS[Non-Max<br/>Suppression]
    NMS --> OUT{Detections}
    OUT --> D1[🔴 Dent + bbox + conf]
    OUT --> D2[🔵 Hole + bbox + conf]
    OUT --> D3[🟢 Deframe + bbox + conf]

    style IMG fill:#34495E,color:#fff
    style YOLO fill:#16A085,color:#fff
    style OUT fill:#C0392B,color:#fff
```



---

## 🗂️ Dataset

- **Source:** [Roboflow](https://roboflow.com) → mirrored on [Kaggle](https://www.kaggle.com) as `siirrii/damaged-shipping-containers`
- **License:** CC BY 4.0
- **Format:** YOLO object-detection format (`images/` + `labels/` per split)

```
Container damaged parts detection dataset/
├── data.yaml
├── train/
│   ├── images/   (544 images)
│   └── labels/   (544 files)
├── valid/
│   ├── images/   (68 images)
│   └── labels/   (68 files)
└── test/
    ├── images/   (68 images)
    └── labels/   (68 files)
```

| Metric | Value |
|---|---|
| Total images | **680** |
| Total annotations | **1,040** bounding boxes |
| Split ratio | 80% train / 10% valid / 10% test |
| Image format | 100% JPEG, RGB (3-channel) |
| Resolutions | 640×640 (612 imgs), 1024×1024 (68 imgs) — all square (AR = 1.0) |
| Integrity | ✅ 0 missing labels · 0 orphan labels · 0 invalid annotations · 0 empty files |
| Instances / image | mean **1.53**, median 1, max 5 |

### Class Balance

| Damage Type | Annotations | Share |
|---|---:|---:|
| 🔴 Dent | 630 | 60.6% |
| 🔵 Hole | 250 | 24.0% |
| 🟢 Deframe | 160 | 15.4% |

> ⚠️ **Class imbalance noted** — `Deframe` is the minority class (15.4%). This drives the per-class evaluation strategy throughout the notebook rather than relying on aggregate mAP alone.

---

## 🔬 Exploratory Data Analysis

<table>
<tr>
<td width="50%">

**Dataset statistics (format / resolution / aspect ratio)**
<img src="assets/assets/phase05_dataset_stats.png" width="100%">

</td>
<td width="50%">

**Class distribution (counts + proportion)**
<img src="assets/assets/phase06_class_distribution.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**Class distribution across splits**
<img src="assets/assets/phase06_split_class_distribution.png" width="100%">

</td>
<td width="50%">

**Bounding-box geometry (width / height / area)**
<img src="assets/assets/phase07_bbox_geometry.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**Object size category (small / medium / large)**
<img src="assets/assets/phase07_size_category.png" width="100%">

</td>
<td width="50%">

**Spatial density heatmap per class**
<img src="assets/assets/phase07_spatial_heatmap.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**Instances per image**
<img src="assets/assets/phase08_instances_per_image.png" width="100%">

</td>
<td width="50%">

**Ground-truth samples with bounding boxes**
<img src="assets/assets/phase09_sample_gt_boxes.png" width="100%">

</td>
</tr>
</table>

**Augmentation preview** (illustrative — flip / HSV jitter / rotation / blur):

<img src="assets/assets/phase10_augmentation_preview.png" width="100%">

---

## 🧠 Model Training

Three YOLO11 variants were trained **under an identical protocol** — same data split, image size, epoch budget, and random seed — to cleanly isolate the effect of model capacity.

```mermaid
gantt
    title Training Configuration Timeline
    dateFormat X
    axisFormat %s
    section YOLO11n
    Nano backbone, 2.59M params      :a1, 0, 100
    section YOLO11s
    Small backbone, 9.43M params     :a2, 0, 100
    section YOLO11m
    Medium backbone, 20.06M params   :a3, 0, 100
```

| Config | Value |
|---|---|
| Image size | 640 × 640 |
| Batch size | 16 |
| Epochs | 100 (early-stop patience 30) |
| Optimizer | auto |
| Seed | 42 |

**Training curves** (validation box loss & mAP50-95 over epochs, all 3 models overlaid):

<img src="assets/assets/phase17_training_curves.png" width="100%">

---

## 📈 Results & Evaluation

### Overall Metrics

<img src="assets/assets/phase17_overall_metrics_comparison.png" width="100%">

### Per-Class AP@50

<img src="assets/assets/phase17_per_class_ap50_comparison.png" width="100%">

| Model | AP50 Dent | AP50 Hole | AP50 Deframe | AP50-95 Dent | AP50-95 Hole | AP50-95 Deframe |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO11n | 0.935 | 0.954 | 0.980 | 0.763 | 0.813 | 0.896 |
| YOLO11s | 0.925 | 0.959 | **0.990** | 0.767 | 0.821 | **0.947** |
| YOLO11m | 0.917 | 0.961 | 0.977 | 0.734 | 0.780 | 0.850 |

### Accuracy vs. Efficiency Frontier

<img src="assets/assets/phase17_efficiency_frontier.png" width="100%">

> YOLO11s sits at the sweet spot — near-YOLO11m accuracy at less than half the parameter count and 3× faster inference than YOLO11m.

### Confusion Matrices (Test Set)

<img src="assets/assets/phase18_confusion_matrices.png" width="100%">

### Precision–Recall Curves (Test Set)

<img src="assets/assets/phase19_pr_curves.png" width="100%">

### Qualitative Inference — Side-by-Side

<img src="assets/assets/phase20_qualitative_comparison.png" width="100%">

### Error Analysis

<img src="assets/assets/phase21_error_analysis.png" width="100%">

| Model | Class | Precision | Recall | Missed-Detection (1-Recall) | False-Positive (1-Precision) |
|---|---|:---:|:---:|:---:|:---:|
| YOLO11n | Dent | 0.901 | 0.875 | 0.125 | 0.099 |
| YOLO11n | Hole | 0.948 | 0.944 | 0.056 | 0.052 |
| YOLO11n | Deframe | 0.949 | **1.000** | **0.000** | 0.051 |
| YOLO11s | Dent | 0.991 | 0.875 | 0.125 | 0.009 |
| YOLO11s | Hole | 0.985 | 0.944 | 0.056 | 0.016 |
| YOLO11s | Deframe | 0.941 | **1.000** | **0.000** | 0.059 |
| YOLO11m | Dent | **1.000** | 0.813 | 0.187 | **0.000** |
| YOLO11m | Hole | 0.991 | 0.944 | 0.056 | 0.009 |
| YOLO11m | Deframe | 0.925 | 0.947 | 0.053 | 0.075 |

**Key finding:** `Dent` recall is the weakest link across *all three* models (81–88%), despite being the majority class — likely due to its highly variable box size (1.9%–87% of image width). `Deframe`, despite being the minority class, achieves perfect recall in the smaller models, suggesting its visual signature is more distinctive than expected.

---

## 📁 Repository Structure

```
container-damage-yolo/
├── README.md
├── damaged.ipynb                  # Full end-to-end notebook (this pipeline)
├── assets/                        # Exported charts used in this README
│   ├── phase05_dataset_stats.png
│   ├── phase06_class_distribution.png
│   ├── phase07_bbox_geometry.png
│   ├── phase09_sample_gt_boxes.png
│   ├── phase17_overall_metrics_comparison.png
│   ├── phase18_confusion_matrices.png
│   └── ...
├── runs/
│   ├── train/
│   │   ├── yolo11n_container_damage/weights/best.pt
│   │   ├── yolo11s_container_damage/weights/best.pt
│   │   └── yolo11m_container_damage/weights/best.pt
│   └── val/
├── results_export/
│   ├── full_metrics_comparison.csv
│   ├── per_class_error_analysis.csv
│   ├── dataset_integrity_report.csv
│   └── class_distribution_summary.csv
├── data.yaml
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/SiriNandinii/container-damage-yolo.git
cd container-damage-yolo
pip install -r requirements.txt
```

`requirements.txt`:
```
ultralytics==8.3.*
opencv-python-headless
pandas
numpy
matplotlib
seaborn
pillow
pyyaml
scikit-learn
tqdm
plotly
kaleido
albumentations
```

### 2. Get the dataset

```bash
# via Kaggle CLI
kaggle datasets download -d siirrii/damaged-shipping-containers -p ./data --unzip
```

### 3. Run the notebook

```bash
jupyter notebook damaged.ipynb
```

Update `DATASET_ROOT` in the dataset-configuration cell to point at your local copy, then run all cells top-to-bottom.

### 4. Run inference with a trained model

```python
from ultralytics import YOLO

model = YOLO("runs/train/yolo11s_container_damage/weights/best.pt")
results = model.predict("path/to/container.jpg", conf=0.25)
results[0].show()
```

---

## 📓 Notebook Phases

<details>
<summary><b>Click to expand the full 24-phase breakdown</b></summary>

| # | Phase |
|---|---|
| 1 | Environment setup — install libraries |
| 2 | Import libraries & global config |
| 3 | Dataset configuration & `data.yaml` |
| 4 | Dataset integrity checks |
| 5 | Basic dataset statistics (format, channels, resolution) |
| 6 | Class distribution analysis |
| 7 | Bounding-box geometry analysis |
| 8 | Instances-per-image analysis |
| 9 | Sample visualization with ground-truth boxes |
| 10 | Augmentation preview |
| 11 | Training configuration |
| 12 | Train YOLO11n |
| 13 | Train YOLO11s |
| 14 | Train YOLO11m |
| 15 | Validation of all 3 models |
| 16 | Metrics aggregation |
| 17 | Comparative visualization (bar / radar / efficiency-frontier) |
| 18 | Confusion matrices |
| 19 | PR curves |
| 20 | Qualitative inference comparison |
| 21 | Error analysis |
| 22 | Final leaderboard & model selection |
| 23 | Export results |
| 24 | Conclusion |

</details>

---

## 🛣️ Roadmap

- [ ] Address `Dent` recall gap with size-aware augmentation or anchor tuning
- [ ] Expand `Deframe` examples via targeted data collection
- [ ] Export best model (`YOLO11s`) to ONNX / TensorRT for edge deployment
- [ ] Build a lightweight inference API (FastAPI) around `best.pt`
- [ ] Add active-learning loop for hard-negative mining on false positives

---


## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If this project helped you, consider starring the repo! ⭐**

Made with 🧡 by [SiriNandinii](https://github.com/SiriNandinii)

</div>
