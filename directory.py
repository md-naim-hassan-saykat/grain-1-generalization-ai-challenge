import os
import shutil
import random

SRC_DIR = "Data/Grain-Data-RGB"
DST_DIR = "Competition_Bundle/input_data"
TRAIN_RATIO = 0.8

# 1. Lister les fichiers
files = [
    f for f in os.listdir(SRC_DIR)
    if os.path.isfile(os.path.join(SRC_DIR, f))
]

# 2. Mélanger
random.shuffle(files)

# 3. Split
split_idx = int(len(files) * TRAIN_RATIO)
train_files = files[:split_idx]
test_files  = files[split_idx:]

# 4. Créer les dossiers
os.makedirs(os.path.join(DST_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DST_DIR, "test"), exist_ok=True)

# 5. Copier (ou déplacer)
for f in train_files:
    shutil.copy2(
        os.path.join(SRC_DIR, f),
        os.path.join(DST_DIR, "train", f)
    )

for f in test_files:
    shutil.copy2(
        os.path.join(SRC_DIR, f),
        os.path.join(DST_DIR, "test", f)
    )

print(f"Train: {len(train_files)} fichiers")
print(f"Test : {len(test_files)} fichiers")
