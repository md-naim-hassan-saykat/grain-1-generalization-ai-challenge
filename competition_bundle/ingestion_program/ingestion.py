import os
import sys
import numpy as np

# DO NOT CHANGE THESE PATHS (Codabench standard)
INPUT_DIR = "/app/input_data"
OUTPUT_DIR = "/app/output"
SUBMISSION_DIR = "/app/ingested_program"

os.makedirs(OUTPUT_DIR, exist_ok=True)
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "prediction")


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


def npz_to_xy(npz_path):
    """
    Load one .npz and return (x, y).
    Priority:
      X: 'features' -> 'x' -> 'X' -> first array
      y: 'label' -> 'y' -> 'Y' -> None
    """
    z = np.load(npz_path, allow_pickle=True)

    # X
    if "features" in z:
        x = z["features"]
    elif "x" in z:
        x = z["x"]
    elif "X" in z:
        x = z["X"]
    else:
        x = z[list(z.files)[0]]

    # y
    if "label" in z:
        y = z["label"]
    elif "y" in z:
        y = z["y"]
    elif "Y" in z:
        y = z["Y"]
    else:
        y = None

    x = np.asarray(x)

    # Flatten feature tensor -> 1D feature vector
    x = x.reshape(-1)

    if y is not None:
        y = np.asarray(y)
        if y.size == 1:
            y = y.reshape(-1)[0]
        y = str(y)  # keep labels as string (safe for Codabench)

    return x, y


def load_split(split_name):
    """
    Try to load split from:
      INPUT_DIR/<split_name>/input_data/*.npz
    If not found, return empty.
    """
    split_dir = os.path.join(INPUT_DIR, split_name, "input_data")
    if not os.path.isdir(split_dir):
        return None, None, []

    files = find_npz_files(split_dir)
    if not files:
        return None, None, []

    X_list, y_list = [], []
    has_labels = False

    for fp in files:
        x, y = npz_to_xy(fp)
        X_list.append(x)
        y_list.append(y)
        if y is not None:
            has_labels = True

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=object) if has_labels else None
    return X, y, files


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
            "Make sure submission.zip contains model.py at the root, "
            "and it defines a class named Model."
        ) from e

    model = Model()

    # 2) Load splits (preferred)
    X_train, y_train, train_files = load_split("train")
    X_val, y_val, val_files = load_split("val")
    X_test, y_test, test_files = load_split("test")

    # 3) Fallback: if no split folders exist, search everything
    if test_files == [] and train_files == [] and val_files == []:
        # Treat everything under INPUT_DIR as test
        all_files = find_npz_files(INPUT_DIR)
        if not all_files:
            raise FileNotFoundError(f"No .npz files found under {INPUT_DIR}")

        X_list = []
        for fp in all_files:
            x, _ = npz_to_xy(fp)
            X_list.append(x)

        X_test = np.stack(X_list, axis=0)
        test_files = all_files

    # 4) Train if labels exist in train split
    if X_train is not None and y_train is not None and len(train_files) > 0:
        y_train = np.asarray(y_train).reshape(-1).astype(str)
        model.train(X_train, y_train)

    # Optional fallback: train on val if train is missing but val has labels
    elif X_val is not None and y_val is not None and len(val_files) > 0:
        y_val = np.asarray(y_val).reshape(-1).astype(str)
        model.train(X_val, y_val)

    else:
        print("Warning: No labeled train/val data found. Running predict() without training.")

    # 5) Predict on test
    if X_test is None or len(test_files) == 0:
        raise ValueError("No test data found to run predictions.")

    preds = model.predict(X_test)
    preds = np.asarray(preds).reshape(-1).astype(str)

    # Safety check: must output exactly 1 prediction per test sample
    if len(preds) != len(test_files):
        raise ValueError(
            f"Predictions length {len(preds)} does not match "
            f"number of test samples {len(test_files)}"
        )

    # 6) Write predictions (one per line) to /app/output/prediction
    with open(PREDICTION_PATH, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(f"{p}\n")

    print("Ingestion completed successfully.")
    print("Num test samples:", len(test_files))
    print("Prediction file written to:", PREDICTION_PATH)

if __name__ == "__main__":
    main()
