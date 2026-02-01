# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# BASELINE VERSION - Uses scikit-learn instead of TensorFlow
# This is a proper baseline that actually learns from the images

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier
        """
        print("[*] - Initializing Baseline Classifier (scikit-learn, no TensorFlow)")
        self.model = None
        self.scaler = StandardScaler()
        self.num_classes = None
        self.img_size = (64, 64)  # Resize to smaller size for speed

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
        print("[*] - Training Baseline Classifier on the train set")
        
        # Get data
        if 'X' in train_data:
            # Data already loaded
            X_train = train_data['X']
            y_train = train_data['y']
        elif 'filepaths' in train_data:
            # Load data from files
            filepaths = train_data['filepaths']
            y_train = train_data['y']
            
            print(f"[*] Loading {len(filepaths)} training images...")
            X_train = []
            for i, filepath in enumerate(filepaths):
                if i % 1000 == 0:
                    print(f"  Loaded {i}/{len(filepaths)} images...")
                data = np.load(filepath)
                img = data['x']
                data.close()
                
                # Resize if needed (simple resize by taking center crop and resizing)
                if img.shape[:2] != self.img_size:
                    # Simple resize: take center and resize
                    h, w = img.shape[:2]
                    target_h, target_w = self.img_size
                    start_h = (h - target_h) // 2
                    start_w = (w - target_w) // 2
                    img = img[start_h:start_h+target_h, start_w:start_w+target_w]
                
                # Flatten image to 1D vector
                img_flat = img.flatten()
                X_train.append(img_flat)
            
            X_train = np.array(X_train, dtype=np.float32)
            print(f"[*] Loaded {len(X_train)} images, shape: {X_train.shape}")
        else:
            raise ValueError("train_data must contain 'X' or 'filepaths'")
        
        # Normalize to [0, 1]
        if X_train.max() > 1.0:
            X_train = X_train / 255.0
        
        # Determine number of classes
        self.num_classes = len(np.unique(y_train))
        print(f"[*] Number of classes: {self.num_classes}")
        
        # Flatten images if not already flattened
        if len(X_train.shape) > 2:
            n_samples = X_train.shape[0]
            X_train = X_train.reshape(n_samples, -1)
        
        # Scale features
        print("[*] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest (simple but effective baseline)
        print("[*] Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=50,  # Small number for speed
            max_depth=10,
            random_state=42,
            n_jobs=-1,  # Use all CPUs
            verbose=1
        )
        self.model.fit(X_train_scaled, y_train)
        print("[*] Training completed")

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
        print("[*] - Predicting test set using Baseline Classifier")
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get data
        if 'X' in test_data:
            X_test = test_data['X']
        elif 'filepaths' in test_data:
            filepaths = test_data['filepaths']
            
            print(f"[*] Loading {len(filepaths)} test images...")
            X_test = []
            for i, filepath in enumerate(filepaths):
                if i % 1000 == 0:
                    print(f"  Loaded {i}/{len(filepaths)} images...")
                data = np.load(filepath)
                img = data['x']
                data.close()
                
                # Resize if needed
                if img.shape[:2] != self.img_size:
                    h, w = img.shape[:2]
                    target_h, target_w = self.img_size
                    start_h = (h - target_h) // 2
                    start_w = (w - target_w) // 2
                    img = img[start_h:start_h+target_h, start_w:start_w+target_w]
                
                # Flatten image
                img_flat = img.flatten()
                X_test.append(img_flat)
            
            X_test = np.array(X_test, dtype=np.float32)
        else:
            raise ValueError("test_data must contain 'X' or 'filepaths'")
        
        # Normalize to [0, 1]
        if X_test.max() > 1.0:
            X_test = X_test / 255.0
        
        # Flatten if needed
        if len(X_test.shape) > 2:
            n_samples = X_test.shape[0]
            X_test = X_test.reshape(n_samples, -1)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        print("[*] Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"[*] Generated {len(y_pred)} predictions")
        print(f"[*] Unique predicted classes: {np.unique(y_pred)}")
        
        return y_pred.astype(np.int32)
