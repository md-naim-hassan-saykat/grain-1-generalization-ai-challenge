import os
import sys
import json
import numpy as np

# Codabench standard paths (do not change)
INPUT_DIR = "/app/input_data"
OUTPUT_DIR = "/app/output"
SUBMISSION_DIR = "/app/ingested_program"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_JSON_PATH = os.path.join(OUTPUT_DIR, "result.json")


# -------------------------
# Helpers
# -------------------------
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


# -------------------------
# Main ingestion logic
# -------------------------
def main():
    # 1) Import participant model
    sys.path.insert(0, SUBMISSION_DIR)
    try:
        from model import Model
    except Exception as e:
        raise ImportError(
            "Could not import Model from submission. "
            "submission.zip must contain model.py at the root, defining class Model."
        ) from e

    model = Model()

    # 2) Locate npz files
    npz_files = find_npz_files(INPUT_DIR)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {INPUT_DIR}")

    # 3) Load data matrix
    X_list = [load_npz_x(fp) for fp in npz_files]
    X = np.stack(X_list, axis=0)

    # 4) Predict (robust)
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict({"X": X, "filepaths": npz_files})

    preds = np.asarray(preds).reshape(-1)

    if len(preds) != len(npz_files):
        raise ValueError(
            f"Predictions length {len(preds)} does not match number of samples {len(npz_files)}"
        )

    # 5) Build result.json mapping filename -> prediction
    # IMPORTANT: scoring expects keys to be filenames, not full paths
    results = {"predictions": {}}
    for fp, p in zip(npz_files, preds):
        fname = os.path.basename(fp)
        results["predictions"][fname] = int(p) if str(p).isdigit() else str(p)

    # 6) Write result.json to /app/output/
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Ingestion completed successfully.")
    print("Num samples:", len(npz_files))
    print("Wrote:", RESULT_JSON_PATH)


if __name__ == "__main__":
    main()
