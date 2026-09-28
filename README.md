# Historical Forest Map Instance Segmentation via Dual Semi-Supervised Pseudo-Labeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper: **"Enhancing instance segmentation on limited-annotated historical maps via pseudo-labeling"** (Submitted to *Ecological Informatics*).

---

## 📌 Overview

Automatic extraction of historical forest boundaries from scanned map archives is often hindered by cartographic variability and severe label scarcity. This repository provides a unified PyTorch/Ultralytics implementation for semi-supervised instance segmentation using **YOLOv11-seg** augmented with two complementary pseudo-labeling filtering mechanisms:

- **Baseline (`none`):** Fully supervised training using only manual annotations.
- **Strategy-O (`object`):** Inclusive Object-Level Filtering that prioritizes informational diversity to bridge the annotation gap under extreme label scarcity ($N=5$).
- **Strategy-P (`patch`):** Restrictive Patch-Level Filtering that emphasizes high boundary purity as manual supervision scales ($N=10, 20, 40$).

---

## 🛠️ Installation & Setup

### 1. Environment Setup

Clone this repository and set up a Virtual Environment:

git clone [https://github.com/Riceton1024/historical-forest-map-segmentation.git](https://github.com/Riceton1024/historical-forest-map-segmentation.git)
cd historical-forest-map-segmentation

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

---

## 📂 Repository Structure

historical-forest-map-segmentation/
├── demo_data/                # Sample dataset for quick pipeline verification
│   ├── images/               # Full test map images
│   └── labels/               # Corresponding YOLO polygon annotations
├── src/                      # Core modules
│   ├── dataset.py            # Map tiling and label adjustment pipeline
│   ├── pseudo_label.py       # Dual-strategy pseudo-label generators
│   └── metrics.py            # Hungarian matching, AW-IoU, and mAP evaluation
├── data.yaml                 # YOLO dataset configuration
├── train.py                  # Self-training execution script
├── evaluate.py               # Whole-map reconstruction & evaluation runner
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation

---

## 🚀 Execution Guide

### 1. Preprocess & Tile Dataset
Tile high-resolution map archives into sub-images for training:

python src/dataset.py --images-dir demo_data/images --labels-dir demo_data/labels --output-dir data

### 2. Model Training with Strategy Selection
Use `--strategy` to choose between baseline supervised training or pseudo-labeling strategies (`none`, `object`, or `patch`):

# Option A: Baseline Supervised Training (No Pseudo-Labels)
python train.py --strategy none --epochs 100 --batch-size 8

# Option B: Strategy-O (Inclusive Object-Level Pseudo-Labeling)
python train.py --strategy object --epochs 100 --batch-size 8

# Option C: Strategy-P (Restrictive Patch-Level Pseudo-Labeling)
python train.py --strategy patch --epochs 100 --batch-size 8

### 3. Evaluate & Reconstruct Whole Maps
Perform tile-based inference, whole-map stitching, Hungarian matching, and metrics report generation:

python evaluate.py --model results/exp_ssl/weights/best.pt --images-dir demo_data/images --labels-dir demo_data/labels --output-txt results/eval_report.txt

---

## 📊 Citation & Acknowledgment

If you find this codebase or methodology useful in your research, please consider citing:

@article{wu2026enhancing,
  title={Enhancing instance segmentation on limited-annotated historical maps via pseudo-labeling},
  author={Wu, Chen-Huan and M{\"a}der, Patrick and Bernhardt-R{\"o}mermann, Markus},
  journal={Ecological Informatics},
  year={2026}
}

This research was supported by funding from the **German Centre for Integrative Biodiversity Research (iDiv) Halle-Jena-Leipzig** (Flexpool).

---

## 📜 License

This project is released under the [MIT License](LICENSE).