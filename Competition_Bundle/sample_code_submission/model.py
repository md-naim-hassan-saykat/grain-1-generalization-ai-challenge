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
        self.img_size = (128, 128)  # Target image size for resizing
        self.num_classes = None
        self.label_encoder = None

    def fit(self, train_data):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_data: dict
            contains train data and labels
            - 'X': training images (numpy array)
            - 'y': training labels (encoded, numpy array)
            - 'y_original': original labels (optional)

        Returns
        -------
        None
        """
        print("[*] - Training Classifier on the train set")
        
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
        
        # Train model
        print("[*] Starting training...")
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        print("[*] Training completed")

    def _build_model(self, input_shape):
        """
        Build a model using a pre-trained EfficientNetB0 for classification.
        This is a baseline model using transfer learning - participants should improve it!
        
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
            contains test data
            - 'X': test images (numpy array)

        Returns
        -------
        y: 1D numpy array
            predicted labels (encoded)
        """
        print("[*] - Predicting test set using trained Classifier")
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
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
        
        # Convert probabilities to class labels
        y_pred = np.argmax(predictions, axis=1)
        
        print(f"[*] Generated {len(y_pred)} predictions")
        print(f"[*] Unique predicted classes: {np.unique(y_pred)}")
        
        return y_pred
