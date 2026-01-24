import os

input_dir = "/app/input/res"
ref_dir = "/app/input/ref"

prediction_file = os.path.join(input_dir, "prediction")

if not os.path.exists(prediction_file):
    raise FileNotFoundError("prediction file not found")

# Dummy score
score = 1.0

with open("/app/output/scores.txt", "w") as f:
    f.write(f"score:{score}\n")

print("Scoring completed")
