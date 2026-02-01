# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# SIMPLE VERSION - No TensorFlow, just for testing the pipeline
# This model makes random predictions to test that the ingestion/scoring pipeline works

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import os


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier
        """
        print("[*] - Initializing Simple Classifier (no TensorFlow)")
        self.model = None
        self.num_classes = None
        self.label_encoder = None
        self.class_distribution = None

    def fit(self, train_data):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_data: dict
            Can contain either:
            - 'X': training images (numpy array) - for backward compatibility
            - 'y': training labels (encoded, numpy array)
            OR
            - 'filepaths': list of file paths to .npz files
            - 'y': training labels (encoded, numpy array)
        """
        print("[*] - Training Simple Classifier on the train set")
        
        # Get labels
        if 'y' in train_data:
            y_train = train_data['y']
        else:
            raise ValueError("train_data must contain 'y' key with labels")
        
        # Determine number of classes
        self.num_classes = len(np.unique(y_train))
        print(f"[*] Number of classes: {self.num_classes}")
        
        # Simple "training": just remember the class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        self.class_distribution = counts / counts.sum()
        self.most_common_class = unique[np.argmax(counts)]
        
        print(f"[*] Class distribution: {dict(zip(unique, self.class_distribution))}")
        print(f"[*] Most common class: {self.most_common_class}")
        print("[*] Training completed (simple baseline - no actual learning)")

    def predict(self, test_data):
        """
        This function predicts labels on test data.

        Parameters
        ----------
        test_data: dict
            Can contain either:
            - 'X': test images (numpy array) - for backward compatibility
            OR
            - 'filepaths': list of file paths to .npz files

        Returns
        -------
        y: 1D numpy array
            predicted labels (encoded)
        """
        print("[*] - Predicting test set using Simple Classifier")
        
        if self.num_classes is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get number of test samples
        if 'X' in test_data:
            num_samples = test_data['X'].shape[0]
        elif 'filepaths' in test_data:
            num_samples = len(test_data['filepaths'])
        else:
            raise ValueError("test_data must contain 'X' or 'filepaths'")
        
        print(f"[*] Making predictions for {num_samples} test samples")
        
        # Simple prediction: predict the most common class for all samples
        # This is a baseline that should work but won't be accurate
        y_pred = np.full(num_samples, self.most_common_class, dtype=np.int32)
        
        print(f"[*] Generated {len(y_pred)} predictions")
        print(f"[*] All predictions are class {self.most_common_class} (most common class from training)")
        
        return y_pred
