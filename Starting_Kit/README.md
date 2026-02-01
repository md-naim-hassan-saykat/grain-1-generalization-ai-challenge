# Starting Kit – Grain-1 Generalization Challenge

> This starting kit is designed for participants with no prior knowledge of the challenge.

This folder contains the **official starting kit** for the **Grain-1 Generalization AI Challenge**.

The goal of this starting kit is to help participants **quickly understand the problem, data format, evaluation procedure, and submission workflow**, and to provide a **working baseline model** that can be extended.

---

## Challenge Summary

The Grain-1 challenge focuses on **grain variety classification under distribution shift**.
Participants are expected to build models that **generalize well** beyond the training distribution.

This starting kit demonstrates:
- How data is loaded and interpreted
- How a baseline model is trained and evaluated
- How Codabench-compatible submissions are generated

---

## Contents of This Folder

Starting_Kit/
├── grain_1_generalization_starting_kit.ipynb
└── README.md

A small sample dataset is included elsewhere in the repository
(under `competition_bundle/input_data/sample_data/input_data/`)
and is automatically detected by the notebook.

### `grain_1_generalization_starting_kit.ipynb`
The main notebook that:
- Loads grain data from `.npz` files
- Explains the data structure
- Trains a baseline classifier
- Evaluates performance using accuracy
- Automatically generates a valid submission file

### `sample_data/`
A **small sample dataset** provided so that:
- The notebook runs end-to-end without external data
- Participants can test the pipeline before using the full dataset

---

## How to Run the Notebook

### Option 1: Run Locally
1. Open a terminal
2. Install required packages (see the first cells of the notebook)
3. Launch Jupyter Notebook or JupyterLab
4. Open `grain_1_generalization_starting_kit.ipynb`
5. Run all cells from top to bottom

> The notebook includes a **commented installation cell** for required Python packages.

---

### Option 2: Run on Google Colab (Recommended)

You can run the starting kit directly on **Google Colab** without any local setup.

**Open the notebook in Google Colab**:  
https://colab.research.google.com/github/md-naim-hassan-saykat/grain-1-generalization-ai-challenge/blob/main/Starting_Kit/grain_1_generalization_starting_kit.ipynb

Steps:
1. Open the link above
2. Select **Runtime → Run all**
3. Follow the notebook instructions

Running on Colab requires **no local installation** and is recommended for quick experimentation.

---

## Baseline Model

The baseline model included in this starting kit:
- Is intentionally **simple**
- Produces **valid predictions**
- Serves as a **lower-bound reference**

The baseline is **not meant to be competitive**, but rather to:
- Demonstrate the full pipeline
- Validate data loading and evaluation
- Provide a reference point for improvement

Participants are encouraged to:
- Improve feature extraction
- Explore robust and generalization-focused methods
- Compare results against the baseline

---

## Evaluation

- **Metric**: Classification Accuracy
- Accuracy is used because:
  - The task is a closed-set classification problem
  - Classes are approximately balanced
  - All errors are equally penalized

Additional metrics can be explored by participants for analysis purposes.

---

## Submission Format

The notebook automatically generates a **Codabench-compatible submission file**.

This ensures that:
- Submissions follow the expected format
- No manual formatting is required
- Participants can focus on modeling rather than infrastructure

---

## Notes for Participants

- The sample data is **not representative** of final evaluation performance
- Performance on the full dataset may differ significantly
- Carefully read notebook explanations before modifying the code

This starting kit is designed to be **clear, modular, and extensible**.

---

## Support

For questions related to the challenge or starting kit, please contact the challenge organizers.
Support is provided on a best-effort basis during the challenge period.
