# Grain-1 Generalization AI Challenge – Competition Bundle

This folder contains the **Codabench competition bundle** for the  
**Grain-1 Generalization AI Challenge**.

The competition bundle defines:
- how participant submissions are ingested,
- how predictions are evaluated,
- how scores are computed and displayed on the leaderboard.

This bundle is designed to be **directly zipped and uploaded to Codabench**.

---

## Folder Structure

```text
competition_bundle/
├── ingestion_program/      # Ingestion logic (loads data + participant submission)
├── scoring_program/        # Scoring logic (computes evaluation metric)
├── input_data/             # Sample input data (small subset for testing)
├── reference_data/         # Ground-truth labels for evaluation
├── competition.yaml        # Codabench competition configuration
└── README.md               # This file
```

---

## Ingestion Program

The **ingestion program**:
- Receives participant submissions (e.g. `model.py`)
- Loads the test data from `input_data/`
- Runs the participant model on the test data
- Saves predictions in the expected output format

The ingestion program follows the standard **Codabench ingestion API**.

---

## Scoring Program

The **scoring program**:
- Loads participant predictions
- Loads the reference labels from `reference_data/`
- Computes the evaluation metric
- Outputs the final score displayed on the leaderboard

### Evaluation Metric
- **Classification Accuracy**

Accuracy is used because:
- The task is a closed-set classification problem
- Classes are approximately balanced
- All misclassification errors are equally penalized

---

## Input Data

- Data samples are provided as `.npz` files
- Each file contains sensor-derived features for a single grain sample

A **small sample dataset** is included under:

```text
input_data/sample_data/input_data/
```

This allows:
- Local testing of the ingestion and scoring programs
- Validation that the competition bundle runs end-to-end

> The full dataset is intentionally not included in the bundle.

---

## Reference Data

The `reference_data/` folder contains:
- Ground-truth labels corresponding to the test samples
- Files required by the scoring program to compute evaluation metrics

---

## How to Build the Competition Bundle

From the repository root, create a zip file containing the competition bundle:

```bash
cd competition_bundle
zip -r competition_bundle.zip .
```

Upload `competition_bundle.zip` directly to Codabench when creating or updating the competition.

---

## How to Test the Competition

1. Upload the zipped competition bundle to Codabench  
2. Create a sample submission (e.g. `model.py` packaged as `submission.zip`)  
3. Submit the sample submission  
4. Verify that:
   - The ingestion program runs successfully  
   - The scoring program computes a score  
   - The score appears on the leaderboard  

---

## Notes

- This bundle is intentionally lightweight and educational  
- It is designed for an academic challenge and demonstration purposes  
- Participants are encouraged to use the provided **Starting Kit** for the correct submission format  

---

## Credits

**Group 1 – Grain (Generalization)**  
AI-Master Challenge Course · Université Paris-Saclay · 2025–26
