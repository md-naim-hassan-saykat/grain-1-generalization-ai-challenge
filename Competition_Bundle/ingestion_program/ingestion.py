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
        Load the training and testing data.

        Args:
            input_dir (str): Directory containing train/ and test/ subdirectories
        """
        print("[*] Loading Train data")
        
        train_dir = os.path.join(input_dir, "train")
        test_dir = os.path.join(input_dir, "test")
        
        # Load training data
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.npz')]
        print(f"[*] Found {len(train_files)} training files")
        
        X_train = []
        y_train = []
        
        for filename in train_files:
            filepath = os.path.join(train_dir, filename)
            data = np.load(filepath)
            X_train.append(data['x'])
            y_train.append(int(data['y']))
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        
        # Normalize images to [0, 1]
        X_train = X_train / 255.0 if X_train.max() > 1.0 else X_train
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        self.train_data = {
            'X': X_train,
            'y': y_train_encoded,
            'y_original': y_train
        }
        
        print(f"[*] Training data loaded: {X_train.shape[0]} samples, shape {X_train[0].shape}")
        print(f"[*] Number of classes: {len(self.label_encoder.classes_)}")
        
        # Load test data (without labels)
        print("[*] Loading Test data")
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
        print(f"[*] Found {len(test_files)} test files")
        
        X_test = []
        self.test_filenames = []
        
        for filename in sorted(test_files):
            filepath = os.path.join(test_dir, filename)
            data = np.load(filepath)
            X_test.append(data['x'])
            self.test_filenames.append(filename)
        
        X_test = np.array(X_test, dtype=np.float32)
        
        # Normalize images to [0, 1]
        X_test = X_test / 255.0 if X_test.max() > 1.0 else X_test
        
        self.test_data = {
            'X': X_test
        }
        
        print(f"[*] Test data loaded: {X_test.shape[0]} samples, shape {X_test[0].shape}")

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
        
        # Create result dictionary
        # Format: {filename: prediction}
        # Predictions should be in the original label space (not encoded)
        # Convert encoded predictions back to original labels if needed
        predictions_original = self.label_encoder.inverse_transform(self.predictions)
        
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

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, "w") as f:
            f.write(json.dumps(self.ingestion_result, indent=4))
