import os
import sys
import json

def main():
    # Typical: python scoring.py <input_dir> <output_dir> <program_dir?>
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"
    os.makedirs(output_dir, exist_ok=True)

    res_dir = os.path.join(input_dir, "res")
    pred_file = os.path.join(res_dir, "prediction")

    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")

    # Minimal: just count lines, return a dummy score
    with open(pred_file, "r") as f:
        preds = [line.strip() for line in f if line.strip() != ""]

    score = 0.0  # dummy score 

    # Write both common formats (some platforms expect one or the other)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump({"score": score, "n_predictions": len(preds)}, f)

    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        f.write(f"score: {score}\n")
        f.write(f"n_predictions: {len(preds)}\n")

    print(f"[SCORING] Read {len(preds)} predictions from {pred_file}")
    print(f"[SCORING] Wrote scores to {output_dir}")

if __name__ == "__main__":
    main()
