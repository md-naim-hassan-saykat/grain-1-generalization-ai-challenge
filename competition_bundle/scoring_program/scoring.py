import os
import json

# Codabench standard paths
PRED_DIR = "/app/input/res"
REF_DIR = "/app/input/ref"
OUT_DIR = "/app/output"

os.makedirs(OUT_DIR, exist_ok=True)

RESULT_JSON = os.path.join(PRED_DIR, "result.json")
REF_JSON = os.path.join(REF_DIR, "test_labels.json")  # matches your teammate logs

METRICS_FILE = os.path.join(OUT_DIR, "metrics.json")
SCORES_FILE = os.path.join(OUT_DIR, "scores.txt")
SCORES_JSON = os.path.join(OUT_DIR, "scores.json")


def main():
    # 1) Load reference
    if not os.path.exists(REF_JSON):
        raise FileNotFoundError(f"Reference file not found: {REF_JSON}")

    with open(REF_JSON, "r", encoding="utf-8") as f:
        ref = json.load(f)

    # Expecting: {"labels": {"filename.npz": int_label, ...}}
    if "labels" not in ref or not isinstance(ref["labels"], dict):
        raise ValueError("Reference JSON must contain a dict field: labels")

    y_true_map = ref["labels"]

    # 2) Load predictions
    if not os.path.exists(RESULT_JSON):
        raise FileNotFoundError(f"Result file not found: {RESULT_JSON}")

    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Expecting: {"predictions": {"filename.npz": int_label, ...}}
    if "predictions" not in res or not isinstance(res["predictions"], dict):
        raise ValueError("result.json must contain a dict field: predictions")

    y_pred_map = res["predictions"]

    # 3) Compare on common keys (and enforce same size)
    common = sorted(set(y_true_map.keys()) & set(y_pred_map.keys()))
    if len(common) == 0:
        raise ValueError("No common filenames between predictions and ground truth.")

    # If you want strictness (recommended):
    if len(common) != len(y_true_map):
        missing = sorted(set(y_true_map.keys()) - set(y_pred_map.keys()))
        raise ValueError(f"Missing predictions for {len(missing)} samples. Example: {missing[:5]}")

    correct = 0
    for k in common:
        if str(y_pred_map[k]) == str(y_true_map[k]):
            correct += 1
    accuracy = correct / len(common)

    # 4) Write outputs
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "num_samples": len(common)}, f, indent=2)

    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        f.write(f"accuracy:{accuracy}\n")

    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    print("Scoring completed successfully.")
    print("Accuracy:", accuracy)
    print("Num samples:", len(common))


if __name__ == "__main__":
    main()
