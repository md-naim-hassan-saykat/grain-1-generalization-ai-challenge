import os
import json

# Codabench standard paths
PRED_DIR = "/app/input/res"
REF_DIR  = "/app/input/ref"
OUT_DIR  = "/app/output"

os.makedirs(OUT_DIR, exist_ok=True)

RESULT_JSON = os.path.join(PRED_DIR, "result.json")
REF_JSON    = os.path.join(REF_DIR, "test_labels.json")

SCORES_TXT  = os.path.join(OUT_DIR, "scores.txt")
SCORES_JSON = os.path.join(OUT_DIR, "scores.json")


def debug_list_dir(path, title):
    print(f"[*] {title}: {path}")
    print(f"[*] Exists: {os.path.exists(path)}")
    if os.path.exists(path) and os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            kind = "DIR" if os.path.isdir(full) else "FILE"
            size = os.path.getsize(full) if os.path.isfile(full) else "-"
            print(f"    - {name} ({kind}, {size} bytes)")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(ref_path):
    """
    Expected format (typical):
      {
        "labels": { "file1.npz": 3, ... }
      }
    Or sometimes:
      { "file1.npz": 3, ... }
    """
    ref = load_json(ref_path)

    if isinstance(ref, dict) and "labels" in ref and isinstance(ref["labels"], dict):
        return ref["labels"]

    if isinstance(ref, dict):
        # assume direct mapping filename -> label
        return ref

    raise ValueError("Unsupported ground truth format in test_labels.json")


def load_predictions(result_path):
    """
    Expected format:
      {
        "predictions": { "file1.npz": 3, ... }
      }
    """
    res = load_json(result_path)

    if not isinstance(res, dict) or "predictions" not in res:
        raise ValueError("result.json must contain a top-level key 'predictions'")

    preds = res["predictions"]
    if not isinstance(preds, dict):
        raise ValueError("'predictions' in result.json must be a dict: filename -> label")

    return preds


def compute_accuracy(y_true_map, y_pred_map):
    # evaluate on intersection only (Codabench style)
    common = sorted(set(y_true_map.keys()) & set(y_pred_map.keys()))
    if not common:
        raise ValueError("No common filenames between predictions and ground truth.")

    correct = 0
    for k in common:
        t = str(y_true_map[k]).strip()
        p = str(y_pred_map[k]).strip()
        correct += int(p == t)

    return correct / len(common), len(common)


def main():
    print("----------------------------------------------")
    print("Scoring Program started!")
    print("----------------------------------------------")

    # Debug directories
    debug_list_dir(PRED_DIR, "Predictions directory")
    debug_list_dir(REF_DIR,  "Reference directory")
    debug_list_dir(OUT_DIR,  "Output directory")

    # Load files
    print(f"[*] Loading predictions from: {RESULT_JSON}")
    y_pred_map = load_predictions(RESULT_JSON)

    print(f"[*] Loading ground truth from: {REF_JSON}")
    y_true_map = load_ground_truth(REF_JSON)

    # Compute primary metric
    acc, n = compute_accuracy(y_true_map, y_pred_map)

    print("[*] ========================================")
    print("[*] EVALUATION METRICS")
    print("[*] ========================================")
    print(f"[*] Accuracy: {acc:.4f} ({acc*100:.2f}%) [PRIMARY]")
    print(f"[*] Num common samples: {n}")
    print("[*] ========================================")

    # Write leaderboard outputs
    with open(SCORES_TXT, "w", encoding="utf-8") as f:
        f.write(f"accuracy:{acc}\n")

    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "num_samples": n}, f, indent=2)

    print("[*] Wrote:", SCORES_TXT)
    print("[*] Wrote:", SCORES_JSON)
    print("----------------------------------------------")
    print("[✔] Scoring Program executed successfully!")
    print("----------------------------------------------")


if __name__ == "__main__":
    main()
