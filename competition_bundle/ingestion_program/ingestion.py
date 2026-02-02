import os
import sys
import json
import numpy as np

# Codabench standard paths
INPUT_DIR = "/app/input_data"
OUTPUT_DIR = "/app/output"
SUBMISSION_DIR = "/app/ingested_program"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_TXT = os.path.join(OUTPUT_DIR, "prediction")   # optional legacy
RESULT_JSON = os.path.join(OUTPUT_DIR, "result.json")     # REQUIRED by scorer


def find_npz_files(root_dir):
    """Recursively find all .npz files under root_dir."""
    npz_files = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(r, f))
    return sorted(npz_files)


def load_npz_x(npz_path):
    """
    Load one .npz and return x as a 1D vector.
    Tries common keys: features/x/X, else first array.
    """
    z = np.load(npz_path, allow_pickle=True)
    if "features" in z:
        x = z["features"]
    elif "x" in z:
        x = z["x"]
    elif "X" in z:
        x = z["X"]
    else:
        x = z[list(z.files)[0]]
    return np.asarray(x).reshape(-1)


def main():
    # Import participant model
    sys.path.insert(0, SUBMISSION_DIR)
    try:
        from model import Model
    except Exception as e:
        raise ImportError(
            "Could not import Model from submission. "
            "submission.zip must contain model.py at the root with class Model."
        ) from e

    model = Model()

    # Find npz
    npz_files = find_npz_files(INPUT_DIR)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {INPUT_DIR}")

    # Load features
    X = np.stack([load_npz_x(fp) for fp in npz_files], axis=0)

    # Predict (support both predict(X) and predict(dict))
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict({"X": X, "filepaths": npz_files})

    preds = np.asarray(preds).reshape(-1)

    # Safety check
    if len(preds) != len(npz_files):
        raise ValueError(
            f"Predictions length {len(preds)} does not match "
            f"number of samples {len(npz_files)}"
        )

    # Convert to int labels if possible (best)
    # If model returns strings, keep as-is.
    try:
        preds_out = [int(p) for p in preds]
    except Exception:
        preds_out = [str(p) for p in preds]

    # Build filename->label mapping (what scorer expects)
    predictions_dict = {
        os.path.basename(fp): preds_out[i]
        for i, fp in enumerate(npz_files)
    }

    # Write result.json (REQUIRED)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions_dict}, f, indent=2)

    # Also write prediction text file (optional)
    with open(PREDICTION_TXT, "w", encoding="utf-8") as f:
        for p in preds_out:
            f.write(f"{p}\n")

    print("Ingestion completed successfully.")
    print("Num samples:", len(npz_files))
    print("Wrote:", RESULT_JSON)
    print("Wrote:", PREDICTION_TXT)


if __name__ == "__main__":
    main()
