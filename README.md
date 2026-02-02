# Grain-1 Generalization AI Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/md-naim-hassan-saykat/grain-1-generalization-ai-challenge/blob/main/Starting_Kit/grain_1_generalization_starting_kit.ipynb
)

This repository hosts the **official starting kit and benchmark materials** for the  
**Grain-1 Generalization AI Challenge**, developed as part of the **AI-Master Challenge Course (2025–26)** at **Université Paris-Saclay**.

The challenge is designed to evaluate **robustness and generalization** of machine learning models
under **distribution shifts** in grain variety classification.

---

## Challenge Overview

Grain classification models often perform well on curated training data but fail when exposed to
**new acquisition conditions, sensor variations, or unseen distributions**.

### Objective
Given grain samples stored as `.npz` files, participants must build models that:
- Learn discriminative representations
- Generalize effectively to **unseen or shifted data**
- Produce reliable predictions under distribution change

This challenge emphasizes **generalization performance**, not just in-distribution accuracy.

---

## Data Description

- **Input**:  
  Sensor-derived grain samples stored as `.npz` files  
- **Output**:  
  One predicted label per input sample  
- **Sample Data**:  
  A small, lightweight subset is provided in the repository so that all code runs end-to-end
  without access to the full dataset.

> The full dataset is intentionally not distributed to mimic real-world deployment constraints.

---

## Evaluation Metric

The primary evaluation metric is **classification accuracy**.

Accuracy is used because:
- The task is a closed-set classification problem
- Classes are approximately balanced
- All misclassification errors are equally penalized

Additional metrics (e.g. confusion matrix, F1-score) may be explored by participants.

---

## Starting Kit

The **starting kit** provides:
- A fully runnable Jupyter notebook
- A simple yet valid **baseline model**
- Automatic generation of Codabench-compatible submissions
- Clear separation between **sample data**, **dummy/debug mode**, and **real evaluation mode**

## Contents of This Folder

Starting_Kit/
├── grain_1_generalization_starting_kit.ipynb
└── README.md

A small sample dataset is included elsewhere in the repository
(under `competition_bundle/input_data/sample_data/input_data/`)
and is automatically detected by the notebook.

---

## Getting Started

### Option 1: Run Locally
1. Clone this repository
2. Navigate to the `starting_kit/` folder
3. Open `grain_1_generalization_starting_kit.ipynb`
4. Run all cells

> A commented cell is provided to install required Python packages.

### Option 2: Run on Google Colab
A Colab-compatible version of the notebook is provided for quick experimentation:

**Colab link:**  
https://colab.research.google.com/github/md-naim-hassan-saykat/grain-1-generalization-ai-challenge/blob/main/Starting_Kit/grain_1_generalization_starting_kit.ipynb

---

## Baseline Model

The baseline model included in this repository:
- Uses a simple but meaningful machine learning classifier (Logistic Regression)
- Produces valid predictions
- Serves as a lower-bound reference for comparison

Participants are strongly encouraged to:
- Improve feature extraction
- Explore robust learning techniques
- Investigate domain generalization methods

---

## Codabench Integration

This challenge is deployed using **Codabench**.

The repository includes:

competition_bundle/
├── ingestion_program/
├── scoring_program/
├── input_data/
├── reference_data/
├── README.html
└── competition.yaml

These files define the ingestion, scoring, and evaluation workflow used by Codabench.

---

## Team

**Group 1 – Grain (Generalization)**  
AI-Master Challenge Course · Université Paris-Saclay · 2025–26

> For questions regarding the challenge or the starting kit, please contact the group lead at: mdnaimhassansaykat@gmail.com

---

## License & Usage

This repository is intended for **educational and research purposes** within the context of the
AI Challenge Course.  
Reuse or extension for academic work is encouraged with proper attribution.
