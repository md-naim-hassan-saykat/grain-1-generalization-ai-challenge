"""
Helper functions and classes for Grain Variety Classification Challenge
This file contains all the implementation details.
Users should import these classes in the notebook instead of seeing the full code.
"""

import os
import re
import math
import random
import json
import zipfile
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, cohen_kappa_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# ============================================================================
# Utility Functions
# ============================================================================

def plot_class_distribution(metadata):
    """Plot distribution of samples across varieties."""
    if metadata is None:
        print("Metadata not available. Load data first.")
        return
    
    variety_counts = metadata['varietyNumber'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(variety_counts)), variety_counts.values)
    plt.xlabel('Variety Number')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Samples Across Varieties')
    plt.xticks(range(len(variety_counts)), variety_counts.index, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nSample counts per variety:")
    for var, count in variety_counts.items():
        print(f"  Variety {var}: {count} samples")


def plot_metadata_analysis(metadata):
    """Analyze metadata distributions."""
    if metadata is None:
        print("Metadata not available. Load data first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Microplot distribution
    microplot_counts = metadata['microplotID'].value_counts()
    axes[0, 0].barh(range(len(microplot_counts)), microplot_counts.values)
    axes[0, 0].set_yticks(range(len(microplot_counts)))
    axes[0, 0].set_yticklabels(microplot_counts.index, fontsize=8)
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_title('Distribution Across Microplots')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Year distribution
    year_counts = metadata['year'].value_counts().sort_index()
    axes[0, 1].bar(year_counts.index, year_counts.values)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Distribution Across Years')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Month distribution
    month_counts = metadata['month'].value_counts().sort_index()
    axes[1, 0].bar(month_counts.index, month_counts.values)
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Distribution Across Months')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Variety vs Microplot heatmap
    variety_microplot = pd.crosstab(metadata['varietyNumber'], metadata['microplotID'])
    im = axes[1, 1].imshow(variety_microplot.values, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_xlabel('Microplot')
    axes[1, 1].set_ylabel('Variety')
    axes[1, 1].set_title('Variety Distribution Across Microplots')
    axes[1, 1].set_yticks(range(len(variety_microplot.index)))
    axes[1, 1].set_yticklabels(variety_microplot.index)
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Classes
# ============================================================================

class Data:
    def __init__(self, data_dir, dataset_type="RGB"):
        """
        Initialize Data loader.
        
        Args:
            data_dir: Base directory containing the data folders
            dataset_type: "RGB" for Grain-Data-RGB or "Spectral" for Grain-Data
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        
        # Set the specific dataset directory
        if dataset_type == "RGB":
            self.dataset_dir = os.path.join(data_dir, "Grain-Data-RGB")
        elif dataset_type == "Spectral":
            self.dataset_dir = os.path.join(data_dir, "Grain-Data")
        else:
            raise ValueError("dataset_type must be 'RGB' or 'Spectral'")
        
        self.files = None
        self.metadata = None
        
    def load_and_clean_files(self):
        """Load and clean file list, removing system files."""
        files = os.listdir(self.dataset_dir)
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        # Filter only .npz files
        files = [f for f in files if f.endswith('.npz')]
        self.files = files
        return files
    
    def extract_metadata(self, filename):
        """
        Extract metadata from filename.
        
        Filename format: grain{grainID}_x{x}y{y}-var{varietyNumber}_{timestamp}_corr.npz
        """
        grain_match = re.search(r"grain(?P<grainID>\d+)", filename)
        var_match   = re.search(r"var(?P<varietyNumber>\d+)", filename)
        micro_match = re.search(r"x(?P<x>\d+)y(?P<y>\d+)", filename)
        time_match  = re.search(
            r"2x_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<timestamp>\d+)_corr",
            filename
        )

        if not all([grain_match, var_match, micro_match, time_match]):
            return None

        return {
            "grainID": grain_match.group("grainID"),
            "varietyNumber": var_match.group("varietyNumber"),
            "microplotID": f"x{micro_match.group('x')}y{micro_match.group('y')}",
            "year": time_match.group("year"),
            "month": time_match.group("month"),
            "day": time_match.group("day"),
            "timestamp": time_match.group("timestamp"),
            "filename": filename
        }
    
    def load_data(self):
        """
        Load all data files and extract metadata.
        
        Returns:
            tuple: (files_list, metadata_dataframe)
        """
        # Load and clean files
        files = self.load_and_clean_files()
        
        # Extract metadata from filenames
        all_metadata = [self.extract_metadata(file) for file in files]
        all_metadata = [m for m in all_metadata if m is not None]  # Remove None values
        
        # Convert to pandas dataframe
        self.metadata = pd.DataFrame(all_metadata)
        
        print(f"Loaded {len(files)} files from {self.dataset_dir}")
        print(f"Extracted metadata for {len(self.metadata)} files")
        
        return files, self.metadata
    
    def load_single_file(self, filename):
        """
        Load a single .npz file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            dict: Dictionary containing 'x' (image) and 'y' (label) arrays
        """
        filepath = os.path.join(self.dataset_dir, filename)
        return np.load(filepath)
    
    def get_metadata(self):
        """Get the metadata dataframe."""
        if self.metadata is None:
            self.load_data()
        return self.metadata


class Visualize:
    def __init__(self, data: Data):
        """
        Initialize Visualize class.
        
        Args:
            data: Data object containing loaded grain data
        """
        self.data = data
        self.math = math
        self.random = random
    
    def band_brightness_npz(self, cube, k):
        """
        Approximation of the original Spectralon normalization.
        Used for brightness correction of RGB images.
        """
        band = cube[:, :, k]
        # avoid division by zero
        return np.mean(band) if np.mean(band) != 0 else 1.0
    
    def normalize_image(self, img):
        """Normalize image to [0, 1] range for display."""
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img_norm = (img - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img)
        return img_norm
    
    def prepare_rgb_image(self, img):
        """
        Prepare RGB image for display with brightness correction.
        
        Args:
            img: Image array of shape (H, W, 3)
            
        Returns:
            Normalized RGB image ready for display
        """
        # Apply per-band brightness correction
        r = img[:, :, 0] / self.band_brightness_npz(img, 0)
        g = img[:, :, 1] / self.band_brightness_npz(img, 1)
        b = img[:, :, 2] / self.band_brightness_npz(img, 2)
        img_corrected = np.dstack((b, g, r))  # BGR to RGB for matplotlib
        
        # Normalize per image
        return self.normalize_image(img_corrected)
    
    def plot_random_samples(self, n_samples=20, cols=5, figsize_per_image=1.0):
        """
        Plot a grid of random grain samples.
        
        Args:
            n_samples: Number of random samples to display
            cols: Number of columns in the grid
            figsize_per_image: Size multiplier for each image
        """
        if self.data.files is None:
            self.data.load_data()
        
        files = self.data.files
        if len(files) < n_samples:
            n_samples = len(files)
            print(f"Warning: Only {len(files)} files available, showing all.")
        
        sample_files = self.random.sample(files, n_samples)
        rows = self.math.ceil(n_samples / cols)
        
        plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))
        
        for i, filename in enumerate(sample_files, 0):
            data = self.data.load_single_file(filename)
            img = data["x"]
            y = data["y"]
            
            # Prepare image based on dataset type
            if self.data.dataset_type == "RGB":
                img_display = self.prepare_rgb_image(img)
            else:
                # For spectral data, use first 3 channels or convert to RGB
                if img.shape[2] >= 3:
                    img_display = self.prepare_rgb_image(img[:, :, :3])
                else:
                    img_display = self.normalize_image(img[:, :, 0])
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_display)
            plt.title(f"Var: {y}", fontsize=8)
            plt.axis("off")
        
        plt.suptitle(f"Random Grain Samples (n={n_samples})", fontsize=12)
        plt.tight_layout(pad=0.1)
        plt.show()
    
    def plot_by_variety(self, variety_number, n_samples=10, cols=5, figsize_per_image=1.0):
        """
        Plot samples from a specific variety.
        
        Args:
            variety_number: Variety number to display
            n_samples: Maximum number of samples to display
            cols: Number of columns in the grid
            figsize_per_image: Size multiplier for each image
        """
        if self.data.metadata is None:
            self.data.load_data()
        
        # Filter files by variety
        variety_files = self.data.metadata[
            self.data.metadata['varietyNumber'] == str(variety_number)
        ]['filename'].tolist()
        
        if len(variety_files) == 0:
            print(f"No files found for variety {variety_number}")
            return
        
        if len(variety_files) < n_samples:
            n_samples = len(variety_files)
        
        sample_files = self.random.sample(variety_files, n_samples)
        rows = self.math.ceil(n_samples / cols)
        
        plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))
        
        for i, filename in enumerate(sample_files, 0):
            data = self.data.load_single_file(filename)
            img = data["x"]
            y = data["y"]
            
            # Prepare image based on dataset type
            if self.data.dataset_type == "RGB":
                img_display = self.prepare_rgb_image(img)
            else:
                if img.shape[2] >= 3:
                    img_display = self.prepare_rgb_image(img[:, :, :3])
                else:
                    img_display = self.normalize_image(img[:, :, 0])
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_display)
            plt.title(f"Grain: {filename.split('_')[0]}", fontsize=8)
            plt.axis("off")
        
        plt.suptitle(f"Variety {variety_number} Samples (n={n_samples})", fontsize=12)
        plt.tight_layout(pad=0.1)
        plt.show()
    
    def plot_single_image(self, filename=None, index=None):
        """
        Plot a single grain image.
        
        Args:
            filename: Name of the file to display (optional)
            index: Index in the files list to display (optional)
        """
        if self.data.files is None:
            self.data.load_data()
        
        if filename is None:
            if index is None:
                index = 0
            filename = self.data.files[index]
        
        data = self.data.load_single_file(filename)
        img = data["x"]
        y = data["y"]
        
        # Prepare image based on dataset type
        if self.data.dataset_type == "RGB":
            img_display = self.prepare_rgb_image(img)
        else:
            if img.shape[2] >= 3:
                img_display = self.prepare_rgb_image(img[:, :, :3])
            else:
                img_display = self.normalize_image(img[:, :, 0])
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display)
        plt.title(f"File: {filename}\nVariety: {y}\nShape: {img.shape}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def plot_data(self):
        """
        Default visualization: plot random samples.
        """
        self.plot_random_samples()


class Train:
    def __init__(self, data: Data, test_size=0.2, random_state=42, img_size=(128, 128)):
        """
        Initialize Training class.
        
        Args:
            data: Data object containing loaded grain data
            test_size: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            img_size: Target image size (width, height) for resizing
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.history = None
        
        # Prepare data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, max_samples=None):
        """
        Load and prepare data for training.
        
        Args:
            max_samples: Maximum number of samples to use (None for all)
        """
        if self.data.files is None:
            self.data.load_data()
        
        files = self.data.files
        if max_samples and len(files) > max_samples:
            random.seed(self.random_state)
            files = random.sample(files, max_samples)
        
        print(f"Loading {len(files)} samples...")
        
        X = []
        y = []
        
        for i, filename in enumerate(files):
            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1}/{len(files)} samples...")
            
            data = self.data.load_single_file(filename)
            img = data["x"]
            label = data["y"]
            
            # Resize image if needed
            if img.shape[:2] != self.img_size:
                img_resized = tf.image.resize(img, self.img_size).numpy()
            else:
                img_resized = img
            
            # For RGB dataset, use as is. For spectral, use first 3 channels
            if self.data.dataset_type == "RGB":
                if img_resized.shape[2] == 3:
                    X.append(img_resized)
                else:
                    X.append(img_resized[:, :, :3])
            else:
                # Spectral data - use first 3 channels
                if img_resized.shape[2] >= 3:
                    X.append(img_resized[:, :, :3])
                else:
                    # Pad if less than 3 channels
                    pad = np.zeros((*img_resized.shape[:2], 3 - img_resized.shape[2]))
                    X.append(np.concatenate([img_resized, pad], axis=2))
            
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Normalize images to [0, 1]
        X = X / 255.0 if X.max() > 1.0 else X
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        print(f"\nData prepared:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Image shape: {self.X_train[0].shape}")
        print(f"  Classes: {self.label_encoder.classes_}")
    
    def build_model(self):
        """
        Build a model using a pre-trained EfficientNetB0 for classification.
        This is a baseline model using transfer learning - participants should improve it!
        """
        input_shape = (*self.img_size, 3)
        
        # Load pre-trained EfficientNetB0 (without top layers)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers (optional - can unfreeze for fine-tuning)
        base_model.trainable = False
        
        # Build the model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\nModel architecture:")
        model.summary()
        return model
    
    def train(self, epochs=10, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of training data to use for validation
            verbose: Verbosity level (0, 1, or 2)
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if self.model is None:
            self.build_model()
        
        print(f"\nTraining model for {epochs} epochs...")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose
        )
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Dictionary containing test loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred_classes,
            target_names=[f"Variety {cls}" for cls in self.label_encoder.classes_]
        ))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred_classes
        }
    
    def plot_training_history(self):
        """Plot training history (loss and accuracy)."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class Score:
    def __init__(self, train_obj=None, model=None, X_test=None, y_test=None, label_encoder=None):
        """
        Initialize Score class.
        
        Args:
            train_obj: Train object containing trained model and test data (optional)
            model: Trained Keras model (optional, if train_obj not provided)
            X_test: Test images (optional, if train_obj not provided)
            y_test: Test labels (optional, if train_obj not provided)
            label_encoder: LabelEncoder used for encoding labels (optional)
        """
        if train_obj is not None:
            self.model = train_obj.model
            self.X_test = train_obj.X_test
            self.y_test = train_obj.y_test
            self.label_encoder = train_obj.label_encoder
        else:
            self.model = model
            self.X_test = X_test
            self.y_test = y_test
            self.label_encoder = label_encoder
        
        self.y_pred = None
        self.y_pred_proba = None
        self.scores = {}
        
    def predict(self, verbose=0):
        """
        Generate predictions on test data.
        
        Args:
            verbose: Verbosity level for prediction
        """
        if self.model is None:
            raise ValueError("No model available. Provide a trained model.")
        if self.X_test is None:
            raise ValueError("No test data available.")
        
        print("Generating predictions...")
        self.y_pred_proba = self.model.predict(self.X_test, verbose=verbose)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        print("Predictions generated!")
        return self.y_pred
    
    def compute_score(self, verbose=1):
        """
        Compute comprehensive classification scores.
        
        Args:
            verbose: Verbosity level (0=silent, 1=print results)
            
        Returns:
            dict: Dictionary containing all computed scores
        """
        if self.y_pred is None:
            self.predict(verbose=0)
        
        # Overall accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        self.scores['accuracy'] = accuracy
        
        # Per-class metrics
        class_names = [f"Variety {cls}" for cls in self.label_encoder.classes_] if self.label_encoder else None
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        self.scores['classification_report'] = report
        
        # Extract macro and weighted averages
        self.scores['macro_avg'] = report.get('macro avg', {})
        self.scores['weighted_avg'] = report.get('weighted avg', {})
        
        if verbose >= 1:
            print("=" * 60)
            print("CLASSIFICATION SCORES")
            print("=" * 60)
            print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"\nMacro Average:")
            print(f"  Precision: {self.scores['macro_avg'].get('precision', 0):.4f}")
            print(f"  Recall: {self.scores['macro_avg'].get('recall', 0):.4f}")
            print(f"  F1-score: {self.scores['macro_avg'].get('f1-score', 0):.4f}")
            print(f"\nWeighted Average:")
            print(f"  Precision: {self.scores['weighted_avg'].get('precision', 0):.4f}")
            print(f"  Recall: {self.scores['weighted_avg'].get('recall', 0):.4f}")
            print(f"  F1-score: {self.scores['weighted_avg'].get('f1-score', 0):.4f}")
            print("=" * 60)
        
        return self.scores
    
    def print_classification_report(self):
        """Print detailed classification report."""
        if self.y_pred is None:
            self.predict(verbose=0)
        
        class_names = [f"Variety {cls}" for cls in self.label_encoder.classes_] if self.label_encoder else None
        
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            self.y_test, self.y_pred,
            target_names=class_names,
            zero_division=0
        ))
        print("=" * 60)
    
    def plot_confusion_matrix(self, figsize=(10, 8), normalize=False):
        """
        Plot confusion matrix.
        
        Args:
            figsize: Figure size
            normalize: If True, normalize the confusion matrix
        """
        if self.y_pred is None:
            self.predict(verbose=0)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        
        # Get class names
        if self.label_encoder:
            class_names = [f"Var {cls}" for cls in self.label_encoder.classes_]
        else:
            class_names = [f"Class {i}" for i in range(len(np.unique(self.y_test)))]
        
        # Plot
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def get_per_class_accuracy(self):
        """
        Compute per-class accuracy.
        
        Returns:
            dict: Dictionary mapping class names to their accuracy
        """
        if self.y_pred is None:
            self.predict(verbose=0)
        
        per_class_acc = {}
        unique_classes = np.unique(self.y_test)
        
        for cls in unique_classes:
            mask = self.y_test == cls
            if mask.sum() > 0:
                cls_accuracy = (self.y_pred[mask] == cls).sum() / mask.sum()
                if self.label_encoder:
                    cls_name = f"Variety {self.label_encoder.classes_[cls]}"
                else:
                    cls_name = f"Class {cls}"
                per_class_acc[cls_name] = cls_accuracy
        
        return per_class_acc
    
    def print_per_class_accuracy(self):
        """Print per-class accuracy."""
        per_class_acc = self.get_per_class_accuracy()
        
        print("\n" + "=" * 60)
        print("PER-CLASS ACCURACY")
        print("=" * 60)
        print(f"{'Class':<20} {'Accuracy':<15} {'Samples':<15}")
        print("-" * 60)
        
        for cls_name, acc in per_class_acc.items():
            # Count samples for this class
            if self.label_encoder:
                cls_num = int(cls_name.split()[-1])
                cls_idx = np.where(self.label_encoder.classes_ == str(cls_num))[0]
                if len(cls_idx) > 0:
                    n_samples = (self.y_test == cls_idx[0]).sum()
                else:
                    n_samples = 0
            else:
                n_samples = (self.y_test == int(cls_name.split()[-1])).sum()
            
            print(f"{cls_name:<20} {acc*100:>6.2f}%       {n_samples:>10}")
        print("=" * 60)
    
    def get_top_k_accuracy(self, k=3):
        """
        Compute top-k accuracy.
        
        Args:
            k: Number of top predictions to consider
            
        Returns:
            float: Top-k accuracy
        """
        if self.y_pred_proba is None:
            self.predict(verbose=0)
        
        top_k_pred = np.argsort(self.y_pred_proba, axis=1)[:, -k:]
        top_k_correct = np.array([self.y_test[i] in top_k_pred[i] for i in range(len(self.y_test))])
        top_k_accuracy = top_k_correct.mean()
        
        print(f"\nTop-{k} Accuracy: {top_k_accuracy:.4f} ({top_k_accuracy*100:.2f}%)")
        return top_k_accuracy
    
    def compute_all_metrics(self):
        """
        Compute the 3 key metrics for multi-class classification:
        1. Accuracy: Overall classification accuracy
        2. F1-Score (macro): Balanced metric considering all classes equally
        3. Cohen's Kappa: Agreement beyond chance, accounts for class imbalance
        
        Returns:
            dict: Dictionary containing the 3 computed metrics
        """
        if self.y_pred is None:
            self.predict(verbose=0)
        
        metrics = {}
        
        # 1. Accuracy - Overall classification performance
        metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        
        # 2. F1-Score (macro) - Balanced metric for multi-class problems
        # Macro average treats all classes equally, good for detecting issues with minority classes
        metrics['f1_macro'] = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        
        # 3. Cohen's Kappa - Agreement beyond chance
        # Accounts for class imbalance and chance agreement, more informative than accuracy alone
        metrics['cohen_kappa'] = cohen_kappa_score(self.y_test, self.y_pred)
        
        return metrics
    
    def compute_metrics_with_bootstrap(self, n_bootstrap=100, random_state=42):
        """
        Compute metrics with bootstrap sampling to get error bars.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Dictionary with mean and std for each metric
        """
        if self.y_pred is None:
            self.predict(verbose=0)
        
        np.random.seed(random_state)
        n_samples = len(self.y_test)
        
        # Store all bootstrap results
        bootstrap_metrics = []
        
        print(f"Computing metrics with {n_bootstrap} bootstrap samples...")
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_test_boot = self.y_test[indices]
            y_pred_boot = self.y_pred[indices]
            
            # Compute the 3 key metrics for this bootstrap sample
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test_boot, y_pred_boot)
            metrics['f1_macro'] = f1_score(y_test_boot, y_pred_boot, average='macro', zero_division=0)
            metrics['cohen_kappa'] = cohen_kappa_score(y_test_boot, y_pred_boot)
            
            bootstrap_metrics.append(metrics)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_bootstrap} bootstrap samples...")
        
        # Compute mean and std for each metric
        metric_names = list(bootstrap_metrics[0].keys())
        results = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in bootstrap_metrics]
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        print("✓ Bootstrap computation completed!")
        return results
    
    def plot_metrics_with_error_bars(self, n_bootstrap=100, figsize=(14, 8), random_state=42):
        """
        Plot multiple metrics with error bars using bootstrap.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            figsize: Figure size
            random_state: Random seed for reproducibility
        """
        # Compute metrics with bootstrap
        bootstrap_results = self.compute_metrics_with_bootstrap(n_bootstrap=n_bootstrap, random_state=random_state)
        
        # Prepare data for plotting
        metric_names = []
        means = []
        stds = []
        
        # Select the 3 key metrics to display
        display_metrics = [
            ('accuracy', 'Accuracy', 'Overall classification accuracy'),
            ('f1_macro', 'F1-Score (Macro)', 'Balanced metric for multi-class problems'),
            ('cohen_kappa', "Cohen's Kappa", 'Agreement beyond chance')
        ]
        
        for metric_key, metric_label, metric_desc in display_metrics:
            if metric_key in bootstrap_results:
                metric_names.append(metric_label)
                means.append(bootstrap_results[metric_key]['mean'])
                stds.append(bootstrap_results[metric_key]['std'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(len(metric_names))
        bars = ax.barh(x_pos, means, xerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.01, i, f'{mean:.3f} ± {std:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Key Classification Metrics with 95% Confidence Intervals (Bootstrap)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, min(1.0, max(means) + max(stds) * 2)])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary with explanations
        print("\n" + "=" * 80)
        print("KEY METRICS SUMMARY (with Bootstrap Confidence Intervals)")
        print("=" * 80)
        print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'95% CI':<20} {'Description'}")
        print("-" * 80)
        
        metric_descriptions = {
            'Accuracy': 'Overall classification accuracy',
            'F1-Score (Macro)': 'Balanced metric for multi-class (treats all classes equally)',
            "Cohen's Kappa": 'Agreement beyond chance (accounts for class imbalance)'
        }
        
        for name, mean, std in zip(metric_names, means, stds):
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std
            desc = metric_descriptions.get(name, '')
            print(f"{name:<25} {mean:<12.4f} {std:<12.4f} [{ci_lower:.4f}, {ci_upper:.4f}] {desc}")
        print("=" * 80)
        
        return bootstrap_results


class Submission:
    def __init__(self, submission_dir="./submission", zip_file_name=None):
        """
        Initialize Submission class.
        
        Args:
            submission_dir: Directory where submission files will be saved
            zip_file_name: Name of the zip file (if None, auto-generates with timestamp)
        """
        self.submission_dir = submission_dir
        os.makedirs(self.submission_dir, exist_ok=True)
        
        if zip_file_name is None:
            zip_file_name = f"Submission_{datetime.datetime.now().strftime('%y-%m-%d-%H-%M')}.zip"
        self.zip_file_name = zip_file_name
        
        self.saved_files = []
    
    def save_code(self, train_obj=None, model=None, model_name="model.h5", metadata=None):
        """
        Save trained model for code submission.
        
        Args:
            train_obj: Train object containing trained model (optional)
            model: Trained Keras model (optional, if train_obj not provided)
            model_name: Name for the saved model file
            metadata: Dictionary with additional metadata to save (optional)
        """
        if train_obj is not None:
            model = train_obj.model
            if metadata is None:
                metadata = {
                    'num_classes': train_obj.num_classes,
                    'img_size': train_obj.img_size,
                    'dataset_type': train_obj.data.dataset_type,
                    'label_encoder_classes': train_obj.label_encoder.classes_.tolist() if train_obj.label_encoder else None
                }
        
        if model is None:
            raise ValueError("No model provided. Provide either train_obj or model.")
        
        # Save model
        model_path = os.path.join(self.submission_dir, model_name)
        model.save(model_path)
        self.saved_files.append(model_name)
        print(f"✓ Model saved to: {model_path}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(self.submission_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.saved_files.append("metadata.json")
            print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path
    
    def save_result(self, train_obj=None, score_obj=None, model=None, X_test=None, 
                   predictions=None, predictions_file="predictions.npy", metadata=None):
        """
        Save predictions for result submission.
        
        Args:
            train_obj: Train object containing model and test data (optional)
            score_obj: Score object containing predictions (optional)
            model: Trained Keras model (optional)
            X_test: Test images (optional)
            predictions: Pre-computed predictions array (optional)
            predictions_file: Name for the predictions file
            metadata: Dictionary with additional metadata to save (optional)
        """
        # Get predictions
        if predictions is not None:
            y_pred = predictions
        elif score_obj is not None:
            if score_obj.y_pred is None:
                score_obj.predict(verbose=0)
            y_pred = score_obj.y_pred
        elif train_obj is not None:
            if train_obj.model is None:
                raise ValueError("Model not trained in train_obj.")
            print("Generating predictions...")
            y_pred_proba = train_obj.model.predict(train_obj.X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        elif model is not None and X_test is not None:
            print("Generating predictions...")
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            raise ValueError("Insufficient information to generate predictions. "
                          "Provide predictions, score_obj, train_obj, or (model + X_test).")
        
        # Save predictions
        predictions_path = os.path.join(self.submission_dir, predictions_file)
        np.save(predictions_path, y_pred)
        self.saved_files.append(predictions_file)
        print(f"✓ Predictions saved to: {predictions_path}")
        print(f"  Shape: {y_pred.shape}")
        print(f"  Unique predictions: {np.unique(y_pred)}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {
                'num_predictions': len(y_pred),
                'num_classes': len(np.unique(y_pred)),
                'predictions_shape': y_pred.shape,
                'timestamp': datetime.datetime.now().isoformat()
            }
            if train_obj is not None:
                metadata['dataset_type'] = train_obj.data.dataset_type
                metadata['img_size'] = train_obj.img_size
                if train_obj.label_encoder:
                    metadata['label_encoder_classes'] = train_obj.label_encoder.classes_.tolist()
        
        # Save metadata
        metadata_path = os.path.join(self.submission_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.saved_files.append("metadata.json")
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return predictions_path
    
    def save_readme(self, content=None):
        """
        Save a README file with submission information.
        
        Args:
            content: Custom README content (if None, generates default)
        """
        if content is None:
            content = f"""# Submission for Grain Variety Classification Challenge

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files included:
{chr(10).join(f'- {f}' for f in self.saved_files)}

## Notes:
- This submission was generated using the baseline starting kit
- Participants should improve upon this baseline for better performance
"""
        
        readme_path = os.path.join(self.submission_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(content)
        self.saved_files.append("README.txt")
        print(f"✓ README saved to: {readme_path}")
        
    def zip_submission(self, include_readme=True):
        """
        Create a ZIP file containing all submission files.
        
        Args:
            include_readme: Whether to include a README file
        """
        if include_readme and "README.txt" not in self.saved_files:
            self.save_readme()
        
        if not self.saved_files:
            print("Warning: No files to zip. Save model or predictions first.")
            return None
        
        # Path to ZIP
        zip_path = os.path.join(self.submission_dir, self.zip_file_name)

        # Create ZIP containing the submission directory files
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in self.saved_files:
                file_path = os.path.join(self.submission_dir, filename)
                if os.path.exists(file_path):
                    zf.write(file_path, arcname=filename)
        
        print(f"\n{'='*60}")
        print(f"✓ Submission ZIP created successfully!")
        print(f"  Location: {zip_path}")
        print(f"  Files included: {len(self.saved_files)}")
        print(f"  Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        return zip_path
    
    def list_files(self):
        """List all files in the submission directory."""
        print(f"\nFiles in submission directory ({self.submission_dir}):")
        print("-" * 60)
        if self.saved_files:
            for f in self.saved_files:
                file_path = os.path.join(self.submission_dir, f)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    size_str = f"{size / 1024:.2f} KB" if size < 1024*1024 else f"{size / (1024*1024):.2f} MB"
                    print(f"  ✓ {f:<30} ({size_str})")
                else:
                    print(f"  ✗ {f:<30} (not found)")
        else:
            print("  (no files saved yet)")
        print("-" * 60)