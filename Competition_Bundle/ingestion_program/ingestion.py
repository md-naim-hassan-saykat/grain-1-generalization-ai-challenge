# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder


class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        None

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * train_data (dict): The train data dict.
        * test_data (dict): The test data dict.
        * ingestion_result (dict): The ingestion result dict.
    """

    def __init__(self):
        """
        Initialize the Ingestion class.

        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.test_filenames = None
        self.ingestion_result = None
        self.label_encoder = LabelEncoder()
        self.predictions = None

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_train_and_test_data(self, input_dir):
        """
        Load metadata (file paths and labels) instead of loading all data into memory.
        This prevents memory issues when dealing with large datasets.

        Args:
            input_dir (str): Directory containing train/ and test/ subdirectories
        """
        print("[*] Loading Train data metadata")
        
        train_dir = os.path.join(input_dir, "train")
        test_dir = os.path.join(input_dir, "test")
        
        # Get training file paths and labels (without loading images)
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.npz')]
        print(f"[*] Found {len(train_files)} training files")
        
        # Load only labels to determine classes
        y_train = []
        train_filepaths = []
        
        for filename in train_files:
            filepath = os.path.join(train_dir, filename)
            # Load only the label, not the image
            data = np.load(filepath)
            y_train.append(int(data['y']))
            train_filepaths.append(filepath)
            data.close()  # Explicitly close to free memory
        
        y_train = np.array(y_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Store paths instead of loaded data
        self.train_data = {
            'train_dir': train_dir,
            'filepaths': train_filepaths,
            'y': y_train_encoded,
            'y_original': y_train,
            'num_samples': len(train_files)
        }
        
        print(f"[*] Training metadata loaded: {len(train_files)} samples")
        print(f"[*] Number of classes: {len(self.label_encoder.classes_)}")
        
        # Get test file paths (without loading images)
        print("[*] Loading Test data metadata")
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
        print(f"[*] Found {len(test_files)} test files")
        
        self.test_filenames = sorted(test_files)
        test_filepaths = [os.path.join(test_dir, filename) for filename in self.test_filenames]
        
        self.test_data = {
            'test_dir': test_dir,
            'filepaths': test_filepaths,
            'num_samples': len(test_files)
        }
        
        print(f"[*] Test metadata loaded: {len(test_files)} samples")

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        print("[*] Initializing Submitted Model")
        # Initialize the model from submission
        self.model = Model()
        print("[*] Model initialized successfully")

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        print("[*] Fitting Submitted Model")
        if self.train_data is None:
            raise ValueError("Training data not loaded. Call load_train_and_test_data() first.")
        
        # Train the model with training data
        self.model.fit(self.train_data)
        print("[*] Model training completed")

    def predict_submission(self):
        """
        Make predictions using the submitted model.
        """
        print("[*] Calling predict method of submitted model")
        
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_train_and_test_data() first.")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call init_submission() first.")
        
        # Make predictions on test data
        self.predictions = self.model.predict(self.test_data)
        
        # Convert predictions to integers if needed
        if isinstance(self.predictions, np.ndarray):
            self.predictions = self.predictions.astype(int)
        
        print(f"[*] Predictions generated: {len(self.predictions)} predictions")
        print(f"[*] Prediction shape: {self.predictions.shape}")
        print(f"[*] Unique predicted classes: {np.unique(self.predictions)}")

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        print("[*] Computing Ingestion Result")
        
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call predict_submission() first.")
        
        if self.test_filenames is None:
            raise ValueError("Test filenames not available.")
        
        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized. Call load_train_and_test_data() first.")
        
        print(f"[*] Number of predictions: {len(self.predictions)}")
        print(f"[*] Number of test filenames: {len(self.test_filenames)}")
        print(f"[*] Label encoder classes: {len(self.label_encoder.classes_)}")
        
        # Verify predictions and filenames have same length
        if len(self.predictions) != len(self.test_filenames):
            raise ValueError(
                f"Mismatch: {len(self.predictions)} predictions but {len(self.test_filenames)} filenames"
            )
        
        # Create result dictionary
        # Format: {filename: prediction}
        # Predictions should be in the original label space (not encoded)
        # Convert encoded predictions back to original labels if needed
        try:
            predictions_original = self.label_encoder.inverse_transform(self.predictions)
            print(f"[*] Converted predictions to original label space")
        except Exception as e:
            print(f"[!] ERROR: Failed to inverse transform predictions: {e}")
            print(f"[!] Predictions shape: {self.predictions.shape}")
            print(f"[!] Predictions unique values: {np.unique(self.predictions)}")
            print(f"[!] Label encoder classes: {self.label_encoder.classes_}")
            raise
        
        # Create dictionary mapping filenames to predictions
        result_dict = {}
        for filename, pred in zip(self.test_filenames, predictions_original):
            result_dict[filename] = int(pred)
        
        # Also save as ordered list for easy loading
        self.ingestion_result = {
            "predictions": result_dict,
            "predictions_array": predictions_original.tolist(),
            "filenames": self.test_filenames,
            "num_predictions": len(self.predictions),
            "num_classes": len(self.label_encoder.classes_)
        }
        
        print(f"[*] Ingestion result computed: {len(result_dict)} predictions")
        print(f"[*] Sample predictions (first 3): {list(result_dict.items())[:3]}")

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        if output_dir is None:
            raise ValueError("output_dir must be provided")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        result_file = os.path.join(output_dir, "result.json")
        print(f"[*] Saving result to: {result_file}")
        
        if self.ingestion_result is None:
            raise ValueError("Ingestion result is None. Call compute_result() first.")
        
        try:
            with open(result_file, "w") as f:
                f.write(json.dumps(self.ingestion_result, indent=4))
            
            # Verify file was created
            if os.path.exists(result_file):
                file_size = os.path.getsize(result_file)
                print(f"[*] Result file saved successfully: {result_file} ({file_size} bytes)")
            else:
                raise IOError(f"Result file was not created: {result_file}")
        except Exception as e:
            print(f"[!] ERROR: Failed to save result file: {e}")
            print(f"[!] Output directory: {output_dir}")
            print(f"[!] Output directory exists: {os.path.exists(output_dir)}")
            print(f"[!] Output directory writable: {os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else 'N/A'}")
            raise
