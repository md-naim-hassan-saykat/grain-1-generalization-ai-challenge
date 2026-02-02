import os
import json

# Codabench standard paths
PRED_DIR = "/app/input/res"
REF_DIR = "/app/input/ref"
OUT_DIR = "/app/output"

os.makedirs(OUT_DIR, exist_ok=True)

PREDICTION_FILE = os.path.join(PRED_DIR, "prediction")
REFERENCE_FILE = os.path.join(REF_DIR, "ground_truth")
METRICS_FILE = os.path.join(OUT_DIR, "metrics.json")


def load_labels(path):
    """Load labels from a text file (one label per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # 1) Load predictions and ground truth
    y_pred = load_labels(PREDICTION_FILE)
    y_true = load_labels(REFERENCE_FILE)

    # 2) Safety check
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Prediction length ({len(y_pred)}) does not match "
            f"ground truth length ({len(y_true)})"
        )

    # 3) Compute accuracy
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    # 4) Write metrics.json (Codabench-required)
    metrics = {
        "accuracy": accuracy
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print("Scoring completed successfully.")
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
