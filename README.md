# Grain-1 Generalization AI Challenge

This repository hosts the **official starting kit and benchmark materials** for the  
**Grain-1 Generalization AI Challenge**, developed as part of the **M1 AI Challenge Course (2025–26)**  
at **Université Paris-Saclay**.

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
  A small, lightweight subset is provided in the starting kit so that all code runs end-to-end
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

### Structure

starting_kit/
├── grain_1_generalization_starting_kit.ipynb
├── sample_data/
└── README.md

The notebook is intentionally modular and documented to make extension easy.

---

## Getting Started

### Option 1: Run Locally
1. Clone this repository
2. Navigate to the `starting_kit/` folder
3. Open `grain_1_generalization_starting_kit.ipynb`
4. Run all cells

A commented cell is provided to install required Python packages.

### Option 2: Run on Google Colab
A Colab-compatible version of the notebook is provided for quick experimentation:

**Colab link:** *(add your Colab URL here)*

---

## Baseline Model

The baseline model included in this repository:
- Uses a simple statistical strategy
- Produces valid predictions
- Serves as a **lower-bound reference**, not a competitive solution

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
M1 AI Challenge Course · Université Paris-Saclay · 2025–26

- **Md Naim Hassan Saykat** (Group Lead)  
- *Lubin Longuépée*  
- *Eloi Beurtheret*
- *Lounès Kebdi*  
- *Bill Tang*

For questions regarding the challenge or starting kit, please contact the group lead.

---

## License & Usage

This repository is intended for **educational and research purposes** within the context of the
AI Challenge Course.  
Reuse or extension for academic work is encouraged with proper attribution.
