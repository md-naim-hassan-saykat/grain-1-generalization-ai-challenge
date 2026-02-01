#!/usr/bin/env python3
"""
Script temporaire pour créer un dataset réduit pour tester rapidement sur Codabench.
Ce script crée un échantillon de données train/test pour accélérer les tests.

Usage:
    python3 create_mini_dataset.py          # Crée un mini dataset (100 train, 50 test)
    python3 create_mini_dataset.py --restore  # Restaure les données originales
"""

import os
import shutil
import json
import numpy as np
import random
from pathlib import Path
import argparse

# Configuration
SCRIPT_DIR = Path(__file__).parent
INPUT_DATA_DIR = SCRIPT_DIR / "input_data"
REFERENCE_DATA_DIR = SCRIPT_DIR / "reference_data"
BACKUP_DIR = SCRIPT_DIR / "_backup_full_data"

# Paramètres du mini dataset
MINI_TRAIN_SIZE = 100  # Nombre de fichiers train à garder
MINI_TEST_SIZE = 50    # Nombre de fichiers test à garder


def create_backup():
    """Crée une sauvegarde des données complètes."""
    print("[*] Creating backup of full dataset...")
    
    if BACKUP_DIR.exists():
        print("  [!] Backup already exists, skipping...")
        return
    
    BACKUP_DIR.mkdir(exist_ok=True)
    
    # Backup input_data
    backup_input = BACKUP_DIR / "input_data"
    if INPUT_DATA_DIR.exists():
        shutil.copytree(INPUT_DATA_DIR, backup_input)
        print(f"  ✓ Backed up input_data/")
    
    # Backup reference_data
    backup_ref = BACKUP_DIR / "reference_data"
    if REFERENCE_DATA_DIR.exists():
        shutil.copytree(REFERENCE_DATA_DIR, backup_ref)
        print(f"  ✓ Backed up reference_data/")
    
    print("[*] Backup created successfully!")


def restore_backup():
    """Restaure les données complètes depuis la sauvegarde."""
    print("[*] Restoring full dataset from backup...")
    
    if not BACKUP_DIR.exists():
        print("  [!] No backup found! Cannot restore.")
        return
    
    backup_input = BACKUP_DIR / "input_data"
    backup_ref = BACKUP_DIR / "reference_data"
    
    # Restore input_data
    if backup_input.exists():
        if INPUT_DATA_DIR.exists():
            shutil.rmtree(INPUT_DATA_DIR)
        shutil.copytree(backup_input, INPUT_DATA_DIR)
        print(f"  ✓ Restored input_data/")
    
    # Restore reference_data
    if backup_ref.exists():
        if REFERENCE_DATA_DIR.exists():
            shutil.rmtree(REFERENCE_DATA_DIR)
        shutil.copytree(backup_ref, REFERENCE_DATA_DIR)
        print(f"  ✓ Restored reference_data/")
    
    print("[*] Full dataset restored successfully!")


def create_mini_dataset():
    """Crée un dataset réduit pour les tests."""
    print(f"[*] Creating mini dataset ({MINI_TRAIN_SIZE} train, {MINI_TEST_SIZE} test)...")
    
    # Créer backup d'abord
    create_backup()
    
    train_dir = INPUT_DATA_DIR / "train"
    test_dir = INPUT_DATA_DIR / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print("  [!] Error: input_data/train/ or input_data/test/ not found!")
        return
    
    # === TRAIN DATA ===
    print("\n[*] Processing train data...")
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.npz')])
    print(f"  Found {len(train_files)} training files")
    
    if len(train_files) < MINI_TRAIN_SIZE:
        print(f"  [!] Only {len(train_files)} files available, using all of them")
        selected_train = train_files
    else:
        # Sélectionner un échantillon représentatif (stratifié par variété si possible)
        selected_train = random.sample(train_files, MINI_TRAIN_SIZE)
        print(f"  Selected {len(selected_train)} files")
    
    # Créer un répertoire temporaire pour les nouveaux fichiers train
    temp_train_dir = train_dir.parent / "train_temp"
    temp_train_dir.mkdir(exist_ok=True)
    
    # Copier seulement les fichiers sélectionnés
    for filename in selected_train:
        src = train_dir / filename
        dst = temp_train_dir / filename
        shutil.copy2(src, dst)
    
    # Remplacer l'ancien répertoire
    shutil.rmtree(train_dir)
    shutil.move(temp_train_dir, train_dir)
    print(f"  ✓ Reduced train data to {len(selected_train)} files")
    
    # === TEST DATA ===
    print("\n[*] Processing test data...")
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npz')])
    print(f"  Found {len(test_files)} test files")
    
    if len(test_files) < MINI_TEST_SIZE:
        print(f"  [!] Only {len(test_files)} files available, using all of them")
        selected_test = test_files
    else:
        selected_test = random.sample(test_files, MINI_TEST_SIZE)
        print(f"  Selected {len(selected_test)} files")
    
    # Créer un répertoire temporaire pour les nouveaux fichiers test
    temp_test_dir = test_dir.parent / "test_temp"
    temp_test_dir.mkdir(exist_ok=True)
    
    # Copier seulement les fichiers sélectionnés
    for filename in selected_test:
        src = test_dir / filename
        dst = temp_test_dir / filename
        shutil.copy2(src, dst)
    
    # Remplacer l'ancien répertoire
    shutil.rmtree(test_dir)
    shutil.move(temp_test_dir, test_dir)
    print(f"  ✓ Reduced test data to {len(selected_test)} files")
    
    # === UPDATE REFERENCE DATA ===
    print("\n[*] Updating reference data...")
    ref_test_labels_json = REFERENCE_DATA_DIR / "test_labels.json"
    ref_test_labels_txt = REFERENCE_DATA_DIR / "test_labels.txt"
    ref_test_labels_npy = REFERENCE_DATA_DIR / "test_labels.npy"
    
    if ref_test_labels_json.exists():
        # Charger les labels complets
        with open(ref_test_labels_json, 'r') as f:
            all_labels = json.load(f)
        
        # Garder seulement les labels pour les fichiers test sélectionnés
        mini_labels = {filename: all_labels[filename] for filename in selected_test if filename in all_labels}
        
        # Sauvegarder le nouveau fichier JSON
        with open(ref_test_labels_json, 'w') as f:
            json.dump(mini_labels, f, indent=2)
        print(f"  ✓ Updated test_labels.json ({len(mini_labels)} labels)")
        
        # Mettre à jour le fichier TXT
        if ref_test_labels_txt.exists():
            with open(ref_test_labels_txt, 'w') as f:
                f.write("filename,label\n")
                for filename, label in sorted(mini_labels.items()):
                    f.write(f"{filename},{label}\n")
            print(f"  ✓ Updated test_labels.txt")
        
        # Mettre à jour le fichier NPY (ordre trié)
        if ref_test_labels_npy.exists():
            sorted_filenames = sorted(selected_test)
            labels_array = np.array([mini_labels.get(f, 0) for f in sorted_filenames], dtype=np.int32)
            np.save(ref_test_labels_npy, labels_array)
            print(f"  ✓ Updated test_labels.npy")
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("[*] Mini dataset created successfully!")
    print("="*60)
    print(f"  Train files: {len(selected_train)}")
    print(f"  Test files:  {len(selected_test)}")
    print(f"  Backup saved in: {BACKUP_DIR}")
    print("\n  To restore full dataset, run:")
    print("    python3 create_mini_dataset.py --restore")
    print("="*60)


def create_mini_dataset_custom(train_size, test_size):
    """Version de create_mini_dataset avec tailles personnalisées."""
    global MINI_TRAIN_SIZE, MINI_TEST_SIZE
    original_train = MINI_TRAIN_SIZE
    original_test = MINI_TEST_SIZE
    MINI_TRAIN_SIZE = train_size
    MINI_TEST_SIZE = test_size
    try:
        create_mini_dataset()
    finally:
        MINI_TRAIN_SIZE = original_train
        MINI_TEST_SIZE = original_test


def main():
    parser = argparse.ArgumentParser(
        description="Create a mini dataset for quick testing on Codabench"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the full dataset from backup"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=MINI_TRAIN_SIZE,
        help=f"Number of train files to keep (default: {MINI_TRAIN_SIZE})"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=MINI_TEST_SIZE,
        help=f"Number of test files to keep (default: {MINI_TEST_SIZE})"
    )
    
    args = parser.parse_args()
    
    if args.restore:
        restore_backup()
    else:
        create_mini_dataset_custom(args.train_size, args.test_size)


if __name__ == "__main__":
    # Fixer la seed pour la reproductibilité
    random.seed(42)
    np.random.seed(42)
    main()
