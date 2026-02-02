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
SCORES_FILE = os.path.join(OUT_DIR, "scores.txt")
SCORES_JSON = os.path.join(OUT_DIR, "scores.json")


def load_labels(path):
    """Load labels from a text file (one label per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def main():
    y_pred = load_labels(PREDICTION_FILE)
    y_true = load_labels(REFERENCE_FILE)

    if len(y_true) == 0:
        raise ValueError("Ground truth is empty. Cannot compute accuracy.")

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Prediction length ({len(y_pred)}) does not match "
            f"ground truth length ({len(y_true)})"
        )

    correct = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct / len(y_true)

    # Optional debug metrics
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    # Leaderboard output (common formats)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        f.write(f"accuracy:{accuracy}\n")

    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    print("Scoring completed successfully.")
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
