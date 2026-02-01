# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# Created by: Ihsan Ullah
# Created on: 13 Jan, 2026

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("[*] - Initializing Classifier")
        self.model = None
        self.img_size = (64, 64)  # Smaller size for faster training (baseline)
        self.num_classes = None
        self.label_encoder = None

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
            - 'train_dir': directory containing training files

        Returns
        -------
        None
        """
        print("[*] - Training Classifier on the train set")
        
        # Check if data is provided as arrays or file paths
        if 'X' in train_data:
            # Original format: data already loaded
            X_train = train_data['X']
            y_train = train_data['y']
            
            # Determine number of classes
            self.num_classes = len(np.unique(y_train))
            print(f"[*] Number of classes: {self.num_classes}")
            
            # Resize images if needed
            if X_train.shape[1:3] != self.img_size:
                print(f"[*] Resizing images from {X_train.shape[1:3]} to {self.img_size}")
                X_train_resized = []
                for img in X_train:
                    img_resized = tf.image.resize(img, self.img_size).numpy()
                    X_train_resized.append(img_resized)
                X_train = np.array(X_train_resized, dtype=np.float32)
            
            # Build model
            self.model = self._build_model(X_train.shape[1:])
            
            # Train model (fast baseline: 2 epochs, larger batch)
            print("[*] Starting training (fast baseline mode)...")
            self.model.fit(
                X_train,
                y_train,
                epochs=2,  # Reduced for faster baseline
                batch_size=128,  # Larger batch for speed
                validation_split=0.1,
                verbose=1
            )
        else:
            # New format: load data from file paths by batch
            filepaths = train_data['filepaths']
            y_train = train_data['y']
            
            # Determine number of classes
            self.num_classes = len(np.unique(y_train))
            print(f"[*] Number of classes: {self.num_classes}")
            
            # Load first image to determine shape
            first_data = np.load(filepaths[0])
            first_img = first_data['x']
            first_data.close()
            
            # Resize first image to determine input shape
            if first_img.shape[:2] != self.img_size:
                first_img_resized = tf.image.resize(first_img, self.img_size).numpy()
                input_shape = first_img_resized.shape
            else:
                input_shape = first_img.shape
            
            # Build model
            self.model = self._build_model(input_shape)
            
            # Create data generator for training (larger batch for speed)
            batch_size = 128  # Larger batch for faster training
            train_gen = self._data_generator(filepaths, y_train, batch_size=batch_size, shuffle=True)
            val_gen = self._data_generator(filepaths, y_train, batch_size=batch_size, shuffle=False, validation=True)
            
            # Calculate steps per epoch (90% for training, 10% for validation)
            train_size = int(len(filepaths) * 0.9)
            val_size = len(filepaths) - train_size
            steps_per_epoch = max(1, train_size // batch_size)
            validation_steps = max(1, val_size // batch_size)
            
            # Train model with generator (fast baseline: 2 epochs)
            print("[*] Starting training with data generator (fast baseline mode)...")
            self.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=2,  # Reduced for faster baseline
                validation_data=val_gen,
                validation_steps=validation_steps,
                verbose=1
            )
        
        print("[*] Training completed")

    def _data_generator(self, filepaths, labels, batch_size=128, shuffle=True, validation=False):
        """
        Generator that loads images in batches to avoid memory issues.
        
        Parameters
        ----------
        filepaths: list
            List of file paths to .npz files
        labels: numpy array
            Array of labels
        batch_size: int
            Batch size
        shuffle: bool
            Whether to shuffle data
        validation: bool
            If True, use last 10% for validation
        
        Yields
        ------
        tuple: (batch_images, batch_labels)
        """
        # Create indices array
        all_indices = np.arange(len(filepaths))
        
        if validation:
            # Use last 10% for validation
            split_idx = int(len(all_indices) * 0.9)
            indices = all_indices[split_idx:]
            labels_subset = labels[split_idx:]
        else:
            # Use first 90% for training
            split_idx = int(len(all_indices) * 0.9)
            indices = all_indices[:split_idx]
            labels_subset = labels[:split_idx]
        
        # Create mapping: original index -> position in subset
        # This is needed because indices contains original positions, but labels_subset is indexed from 0
        index_to_position = {orig_idx: pos for pos, orig_idx in enumerate(indices)}
        
        while True:
            # Shuffle indices at the start of each epoch (for training only)
            if shuffle and not validation:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = []
                batch_labels = []
                
                for orig_idx in batch_indices:
                    data = np.load(filepaths[orig_idx])
                    img = data['x']
                    data.close()
                    
                    # Normalize to [0, 1]
                    if img.max() > 1.0:
                        img = img / 255.0
                    
                    # Resize if needed
                    if img.shape[:2] != self.img_size:
                        img = tf.image.resize(img, self.img_size).numpy()
                    
                    batch_images.append(img.astype(np.float32))
                    # Map original index to position in labels_subset
                    pos_in_subset = index_to_position[orig_idx]
                    batch_labels.append(labels_subset[pos_in_subset])
                
                yield np.array(batch_images), np.array(batch_labels)
    
    def _build_model(self, input_shape):
        """
        Build a model using a pre-trained EfficientNetB0 for classification.
        This is a FAST baseline model using transfer learning - participants should improve it!
        
        Optimizations for speed:
        - Small image size (64x64)
        - Only 2 epochs
        - Large batch size (128)
        
        Parameters
        ----------
        input_shape: tuple
            Shape of input images (height, width, channels)
        
        Returns
        -------
        model: keras.Model
            Compiled Keras model
        """
        # Load pre-trained EfficientNetB0 (without top classification layer)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers (optional - can unfreeze for fine-tuning)
        base_model.trainable = False
        
        # Build the complete model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

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
        print("[*] - Predicting test set using trained Classifier")
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Check if data is provided as arrays or file paths
        if 'X' in test_data:
            # Original format: data already loaded
            X_test = test_data['X']
            
            # Resize images if needed
            if X_test.shape[1:3] != self.img_size:
                print(f"[*] Resizing test images from {X_test.shape[1:3]} to {self.img_size}")
                X_test_resized = []
                for img in X_test:
                    img_resized = tf.image.resize(img, self.img_size).numpy()
                    X_test_resized.append(img_resized)
                X_test = np.array(X_test_resized, dtype=np.float32)
            
            # Make predictions
            predictions = self.model.predict(X_test, verbose=0)
        else:
            # New format: load data from file paths by batch
            filepaths = test_data['filepaths']
            batch_size = 128  # Larger batch for faster prediction
            all_predictions = []
            
            print(f"[*] Loading and predicting {len(filepaths)} test images in batches...")
            for i in range(0, len(filepaths), batch_size):
                batch_paths = filepaths[i:i+batch_size]
                batch_images = []
                
                for filepath in batch_paths:
                    data = np.load(filepath)
                    img = data['x']
                    data.close()
                    
                    # Normalize to [0, 1]
                    if img.max() > 1.0:
                        img = img / 255.0
                    
                    # Resize if needed
                    if img.shape[:2] != self.img_size:
                        img = tf.image.resize(img, self.img_size).numpy()
                    
                    batch_images.append(img.astype(np.float32))
                
                batch_images = np.array(batch_images)
                batch_predictions = self.model.predict(batch_images, verbose=0)
                all_predictions.append(batch_predictions)
            
            predictions = np.vstack(all_predictions)
        
        # Convert probabilities to class labels
        y_pred = np.argmax(predictions, axis=1)
        
        print(f"[*] Generated {len(y_pred)} predictions")
        print(f"[*] Unique predicted classes: {np.unique(y_pred)}")
        
        return y_pred
