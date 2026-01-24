import os
import numpy as np

# DO NOT CHANGE THESE PATHS
input_dir = "/app/input_data"
output_dir = "/app/output"
submission_dir = "/app/ingested_program"

os.makedirs(output_dir, exist_ok=True)

# Dummy prediction: one line per input file
prediction_path = os.path.join(output_dir, "prediction")

files = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith(".npz")
])

with open(prediction_path, "w") as f:
    for fname in files:
        f.write("0\n")  # dummy class

print(f"Prediction file written to {prediction_path}")
