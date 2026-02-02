import os
import sys
import numpy as np

# Codabench standard paths
INPUT_DIR = "/app/input_data"
OUTPUT_DIR = "/app/output"
SUBMISSION_DIR = "/app/ingested_program"

os.makedirs(OUTPUT_DIR, exist_ok=True)
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "prediction")


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

    x = np.asarray(x).reshape(-1)
    return x


def main():
    # Import participant model
    sys.path.insert(0, SUBMISSION_DIR)
    try:
        from model import Model
    except Exception as e:
        raise ImportError(
            "Could not import Model from submission. "
            "submission.zip must contain model.py at the root, with class Model."
        ) from e

    model = Model()

    # Find all npz files (works for input_data/sample_data/input_data/)
    npz_files = find_npz_files(INPUT_DIR)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {INPUT_DIR}")

    # Load features into a matrix
    X_list = [load_npz_x(fp) for fp in npz_files]
    X = np.stack(X_list, axis=0)

    # Try prediction in a robust way:
    # 1) predict(X)  (numpy style)
    # 2) predict({"X": X, "filepaths": [...]}) (dict style)
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict({"X": X, "filepaths": npz_files})

    preds = np.asarray(preds).reshape(-1).astype(str)

    if len(preds) != len(npz_files):
        raise ValueError(
            f"Predictions length {len(preds)} does not match "
            f"number of samples {len(npz_files)}"
        )

    # Write predictions: one per line
    with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")

    print("Ingestion completed successfully.")
    print("Num samples:", len(npz_files))
    print("Prediction file written to:", PREDICTION_PATH)


if __name__ == "__main__":
    main()
