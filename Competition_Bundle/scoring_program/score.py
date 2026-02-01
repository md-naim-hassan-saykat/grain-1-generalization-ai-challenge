# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


class Scoring:
    """
    This class is used to compute the scores for the competition.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * reference_data (dict): The reference data.
        * ingestion_result (dict): The ingestion result.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.
    """

    def __init__(self, name=""):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.reference_data = None
        self.ingestion_result = None
        self.ingestion_duration = None
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_reference_data(self, reference_dir):
        """
        Load the reference data (test labels).

        Args:
            reference_dir (str): The reference data directory name.
        """
        print("[*] Reading reference data")
        
        # Try to load from JSON file (preferred format)
        reference_data_file = os.path.join(reference_dir, "test_labels.json")
        
        if os.path.exists(reference_data_file):
            with open(reference_data_file, 'r') as f:
                self.reference_data = json.load(f)
            print(f"[*] Loaded reference data from {reference_data_file}")
            print(f"[*] Number of reference labels: {len(self.reference_data)}")
        else:
            raise FileNotFoundError(
                f"Reference data file not found: {reference_data_file}\n"
                f"Make sure test_labels.json exists in {reference_dir}"
            )

    def load_ingestion_result(self, predictions_dir):
        """
        Load the ingestion result (predictions).

        Args:
            predictions_dir (str): The predictions directory name.
        """
        print("[*] Reading ingestion result")
        print(f"[*] Predictions directory: {predictions_dir}")
        print(f"[*] Predictions directory exists: {os.path.exists(predictions_dir)}")
        
        if os.path.exists(predictions_dir):
            print(f"[*] Contents of predictions directory:")
            for item in os.listdir(predictions_dir):
                item_path = os.path.join(predictions_dir, item)
                item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                item_size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                print(f"    - {item} ({item_type}, {item_size} bytes)")
        else:
            print(f"[!] WARNING: Predictions directory does not exist!")
        
        ingestion_result_file = os.path.join(predictions_dir, "result.json")
        print(f"[*] Looking for result.json at: {ingestion_result_file}")
        
        # Try alternative locations if primary path doesn't exist
        alternative_paths = [
            ingestion_result_file,  # Primary path
            "/app/output/result.json",  # Direct from ingestion output (fallback)
        ]
        
        found_file = None
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                found_file = alt_path
                print(f"[*] Found result.json at: {alt_path}")
                break
            else:
                print(f"[*] Not found at: {alt_path}")
        
        if found_file is None:
            print(f"[!] ERROR: result.json not found in any expected location")
            print(f"[!] Searched paths:")
            for alt_path in alternative_paths:
                print(f"    - {alt_path}: NOT FOUND")
            
            # List what's actually in the directories
            print(f"[!] Contents of {predictions_dir}:")
            if os.path.exists(predictions_dir):
                for item in os.listdir(predictions_dir):
                    print(f"    - {item}")
            else:
                print(f"    Directory does not exist")
            
            print(f"[!] Contents of /app/output/ (if exists):")
            if os.path.exists("/app/output"):
                for item in os.listdir("/app/output"):
                    print(f"    - {item}")
            else:
                print(f"    Directory does not exist")
            
            raise FileNotFoundError(
                f"Ingestion result file not found in any expected location.\n"
                f"Searched: {alternative_paths}\n"
                f"Make sure ingestion completed successfully and created result.json"
            )
        
        ingestion_result_file = found_file
        
        print(f"[*] Found result.json, loading...")
        file_size = os.path.getsize(ingestion_result_file)
        print(f"[*] File size: {file_size} bytes")
        
        try:
            with open(ingestion_result_file, 'r') as f:
                self.ingestion_result = json.load(f)
            print(f"[*] Successfully loaded ingestion result from {ingestion_result_file}")
        except json.JSONDecodeError as e:
            print(f"[!] ERROR: Failed to parse JSON from result.json: {e}")
            print(f"[!] File might be corrupted or empty")
            raise
        
        # Extract predictions
        if "predictions" in self.ingestion_result:
            predictions_dict = self.ingestion_result["predictions"]
            print(f"[*] Number of predictions: {len(predictions_dict)}")
            if len(predictions_dict) > 0:
                print(f"[*] Sample predictions (first 3): {list(predictions_dict.items())[:3]}")
        else:
            print(f"[!] ERROR: Ingestion result does not contain 'predictions' key")
            print(f"[!] Available keys: {list(self.ingestion_result.keys())}")
            raise ValueError("Ingestion result does not contain 'predictions' key")

    def compute_scores(self):
        """
        Compute the scores for the competition.
        """
        print("[*] Computing scores")
        
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Call load_reference_data() first.")
        
        if self.ingestion_result is None:
            raise ValueError("Ingestion result not loaded. Call load_ingestion_result() first.")
        
        # Extract predictions and reference labels
        predictions_dict = self.ingestion_result["predictions"]
        
        # Align predictions with reference labels
        # Get all filenames that are in both predictions and reference
        common_filenames = sorted(set(predictions_dict.keys()) & set(self.reference_data.keys()))
        
        if len(common_filenames) == 0:
            raise ValueError(
                "No common filenames between predictions and reference data. "
                "Check that filenames match."
            )
        
        print(f"[*] Found {len(common_filenames)} common samples")
        
        # Extract aligned predictions and labels
        y_pred = np.array([predictions_dict[f] for f in common_filenames])
        y_true = np.array([self.reference_data[f] for f in common_filenames])
        
        # Compute the 3 key metrics for multi-class classification
        # 1. Accuracy: Overall classification accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # 2. F1-Score (macro): Balanced metric that treats all classes equally
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 3. Cohen's Kappa: Agreement beyond chance, accounts for class imbalance
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        
        # Additional metrics for detailed analysis (optional)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store scores
        # Note: The leaderboard uses "accuracy" as the primary metric (see competition.yaml)
        # But we compute all 3 key metrics for comprehensive evaluation
        self.scores_dict = {
            "accuracy": float(accuracy),  # Primary metric for leaderboard
            "f1_macro": float(f1_macro),  # Secondary metric
            "cohen_kappa": float(cohen_kappa),  # Tertiary metric
            "num_samples": len(common_filenames),
            "num_classes": len(np.unique(y_true)),
            "confusion_matrix": cm.tolist()
        }
        
        print(f"[*] ========================================")
        print(f"[*] EVALUATION METRICS")
        print(f"[*] ========================================")
        print(f"[*] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) [PRIMARY - Used for leaderboard]")
        print(f"[*] F1-Score (Macro): {f1_macro:.4f}")
        print(f"[*] Cohen's Kappa: {cohen_kappa:.4f}")
        print(f"[*] ========================================")

    def write_scores(self, output_dir):

        print("[*] Writing scores")
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))
