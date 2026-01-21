import os
import sys
from pathlib import Path

def main():
    # Typical signature: python ingestion.py <input_dir> <output_dir> <program_dir?>
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/output"

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, "prediction")  # IMPORTANT: no extension

    # Create a simple prediction file.
    # Here: one line per .npz file found in input_dir (recursive). Default label = 0
    npz_files = sorted([str(p) for p in Path(input_dir).rglob("*.npz")])

    with open(pred_path, "w") as f:
        if len(npz_files) == 0:
            # fallback: still create a valid file (at least one line)
            f.write("0\n")
        else:
            for _ in npz_files:
                f.write("0\n")

    print(f"[INGESTION] Wrote: {pred_path} ({max(1, len(npz_files))} lines)")

if __name__ == "__main__":
    main()
