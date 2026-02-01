# 🌾 Grain Variety Classification Challenge

**Starting Kit - Complete Guide for Participants**

This repository contains the **starting kit and materials** for **Group 1 (Grain – Generalization)** of the AI-Master Challenge Course (2025–26) at Université Paris-Saclay.

---

## 📋 Table of Contents

1. [Challenge Overview](#challenge-overview)
2. [Understanding the Classification Task](#understanding-the-classification-task)
3. [Data Description](#data-description)
4. [Model Architecture](#model-architecture)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Getting Started](#getting-started)
7. [Repository Structure](#repository-structure)
8. [Team & Contact](#team--contact)

---

## Challenge Overview

### 🎯 Main Objective

The goal of this challenge is to **develop an artificial intelligence model capable of automatically classifying wheat grain varieties** from images. 

**Why is this important?**
- Rapid and accurate identification of grain varieties is crucial for modern agriculture
- It ensures quality, traceability, and compliance of harvests
- Automating this task can significantly improve agricultural process efficiency

### 🔬 Scientific Context

The data comes from grain images captured using two types of imaging techniques:
- **RGB Imaging**: Standard color images (3 channels: Red, Green, Blue)
- **Hyperspectral Imaging**: Images containing many spectral bands (more information than the human eye can perceive)

Each grain was photographed individually, and each image is associated with a known variety (the "label" or "ground truth").

### 🎓 What You Will Learn

By participating in this challenge, you will:
1. **Handle imaging data** for classification
2. **Build and train deep learning models** (CNN with transfer learning)
3. **Evaluate model performance** with multiple metrics
4. **Improve a baseline** to achieve better performance
5. **Understand generalization challenges** in machine learning

---

## Understanding the Classification Task

### 🎯 What We Want to Predict (Target / Labels)

**Question to solve**: "What is the variety of this grain?"

- **Target variable**: The variety number (`varietyNumber`) of each grain
- **Task type**: **Multi-class classification**
  - Unlike binary classification (yes/no), we have multiple possible classes (8 different varieties)
- **Number of classes**: 8 different varieties
- **Format**: Integer representing the variety (e.g., 2, 3, 4, 5, 6, 7, 8)

**Concrete example**:
- You have an image of a grain
- The model must answer: "This grain is variety 7"
- There are 8 possible answers

### 📸 Our Features (Input Variables)

**What the model "sees"**: Individual grain images

- **Data type**: Grain images (like photos)
- **Format**: NumPy arrays (matrices of numbers) of shape `(Height, Width, Channels)`
  - An image is represented as a grid of pixels, where each pixel has values for each color channel
  
  - **RGB Dataset**: Standard color images with 3 channels (Red, Green, Blue) → shape `(252, 252, 3)`
    - Example: An image of 252 pixels high × 252 pixels wide × 3 color channels
  
  - **Spectral Dataset**: Hyperspectral images with multiple channels → shape `(H, W, N_bands)` where N_bands > 3
    - These images contain more information than standard RGB images, capturing details invisible to the human eye

- **Content**: Individual grain images captured with different imaging techniques
  - Each image represents **a single grain** photographed individually

- **Available metadata** (additional information, not used directly as features but useful for analysis):
  - `grainID`: Unique grain identifier (like a serial number)
  - `microplotID`: Coordinates of the microplot where the grain was grown (x, y)
  - `timestamp`: Date and time of image acquisition
  - `year`, `month`, `day`: Detailed temporal information

### 📁 Data Structure

**File format**: Each grain is stored in a `.npz` file (compressed NumPy format)

Each `.npz` file contains:
- **`x`**: The grain image (NumPy array) - **This is our main feature** ⭐
  - This is what the model will analyze to make its prediction
  
- **`y`**: The label/variety (integer) - **This is what we want to predict** 🎯
  - This is the "correct answer" that the model must learn to predict
  
- **`original_filename`**: Original filename (for traceability)
- **`bands`**: Spectral band information (for spectral data only)

**Practical example**:
```python
data = np.load("grain123.npz")
image = data["x"]  # The grain image (what we give to the model)
variety = data["y"]  # The variety (what we want to predict, e.g., 7)
```

### 🎯 Generalization Challenge

**What is generalization?**
Generalization is the ability of a model to work correctly on **new data it has never seen** during training.

**Why is it important?**
A model that works well on training data but fails on new data is useless in practice. This is the "overfitting" problem.

**The challenge**:
The model must be able to correctly classify grains from:
- **Different microplots** (32 different microplots)
  - Growing conditions can vary by location
- **Different acquisition dates** (2020-2021)
  - Lighting, humidity, etc. conditions can vary over time
- **Different imaging conditions**
  - Camera parameters, lighting, etc. can vary

**Your mission**: Improve the baseline so it generalizes better to these variations!

---

## Data Description

### 📦 File Structure

Data is stored in `.npz` files (compressed NumPy format). Each file contains:
- **`x`**: The grain image (NumPy array)
  - RGB Dataset: 3 channels (Red, Green, Blue) → shape `(252, 252, 3)`
  - Spectral Dataset: Multiple channels → shape `(H, W, N_bands)`
- **`y`**: The label/variety (integer) - what we want to predict

### 📝 Metadata in Filenames

Each filename contains important information:
- **`grainID`**: Unique grain identifier (e.g., `grain123`)
- **`varietyNumber`**: Variety number (e.g., `var7`)
- **`microplotID`**: Microplot coordinates (e.g., `x28y21`)
- **`timestamp`**: Date and time of acquisition (e.g., `2020-12-02T135904`)

**Filename format**: `grain{grainID}_x{x}y{y}-var{varietyNumber}_{timestamp}_corr.npz`

### 🎯 Dataset Choice

You can choose between two datasets:

1. **Grain-Data-RGB** (Recommended for beginners)
   - ✅ Standard RGB images (3 channels)
   - ✅ Faster to load and process
   - ✅ Simpler to understand
   - ✅ Sufficient for a good baseline

2. **Grain-Data** (For going further)
   - ✅ Hyperspectral images (multiple channels)
   - ✅ More information available
   - ⚠️ Larger files
   - ⚠️ More complex to process

**💡 Tip**: Start with the RGB dataset, then try the spectral dataset if you want to improve your performance!

### 📊 Sample Data Mode

**For testing purposes**, you can use a limited number of samples to quickly test the pipeline:

- **Sample mode**: Use `max_samples` parameter to limit the number of files loaded (e.g., `max_samples=1000`)
- **Full mode**: Remove `max_samples` parameter to use all available data

**Note**: The full dataset contains ~26,000+ images. Using sample mode is recommended for:
- Quick testing of your code
- Faster iteration during development
- Limited computational resources

Always test with full data before final submission!

---

## Model Architecture

### 🤖 Our Baseline Model

**What is a baseline?**
A baseline is a simple model that serves as a **starting point**. It is not optimized for best performance, but it works and gives you a reference to improve upon.

**Architecture**: Transfer Learning with EfficientNetB0
- We use a pre-trained EfficientNetB0 model (trained on ImageNet) and adapt it to our grain classification task. This is called transfer learning - leveraging knowledge from a large dataset (ImageNet) to solve our specific problem.

**Input**: Images resized to `(128, 128, 3)`
- Why resize? To standardize image size and match EfficientNetB0 input requirements

**Detailed architecture**:
1. **Pre-trained EfficientNetB0 base** (frozen weights from ImageNet)
   - Role: Extract rich visual features learned from millions of images
   - Why frozen? We keep the pre-trained weights fixed and only train the new layers
2. **Global Average Pooling layer**
   - Role: Convert feature maps to a fixed-size vector
3. **Dense layer** (128 units) with ReLU activation
   - Role: Combine extracted features for classification
4. **Dropout** (0.5)
   - Role: Regularization technique to prevent overfitting
5. **Softmax output layer**
   - Role: Convert scores to probabilities for each variety (sum = 100%)

**Training parameters**:
- **Loss function**: `sparse_categorical_crossentropy` (measures prediction error)
- **Optimizer**: Adam (algorithm that adjusts model weights)
- **Metric**: Accuracy - percentage of correct predictions

### 💡 Improvement Suggestions

To beat the baseline, you can try:

1. **Fine-tuning the pre-trained model**:
   - Unfreeze some layers of EfficientNetB0 for fine-tuning
   - Use different pre-trained models (ResNet50, EfficientNetB3, etc.)
   - Try Vision Transformer (ViT) models

2. **Data Augmentation**:
   - Rotation, flip, zoom, brightness adjustment
   - Artificially increases dataset size

3. **Use complete spectral data**:
   - Exploit all spectral bands (not just the first 3)
   - Multi-band fusion techniques

4. **Better preprocessing**:
   - Adaptive normalization
   - Illumination correction techniques
   - Noise reduction

5. **Hyperparameter tuning**:
   - Batch size, learning rate, number of epochs
   - Model architecture (number of layers, filters, etc.)

6. **Advanced techniques**:
   - Learning rate scheduling
   - Early stopping
   - Model ensembles

**💡 Tip**: Start by improving one thing at a time, then combine the best techniques!

---

## Evaluation Metrics

### 📊 The 3 Key Metrics

We use **3 key metrics** well-suited for multi-class classification:

1. **Accuracy**: Overall classification accuracy - simple and interpretable
   - Percentage of correct predictions
   - Primary metric used for the leaderboard on Codabench

2. **F1-Score (Macro)**: Balanced metric that treats all classes equally
   - Harmonic mean of precision and recall
   - Good for detecting issues with minority classes
   - Macro average treats all classes equally

3. **Cohen's Kappa**: Agreement beyond chance
   - Accounts for class imbalance
   - Provides a more informative measure than accuracy alone
   - Measures agreement beyond what would be expected by chance

### 📈 Error Bars with Bootstrap

The evaluation system also supports **bootstrap sampling** to compute confidence intervals:

- **Error bars** show the **95% confidence interval** of each metric
- They help assess the **stability** and **reliability** of your model
- Useful for comparing different models or approaches

**How it works**: The system resamples the test data multiple times (e.g., 100 times) and computes metrics on each sample. This gives us a distribution of metric values, from which we can compute confidence intervals.

---

## Getting Started

### 📚 How to Use the Starting Kit

1. Navigate to the [`Starting_Kit`](Starting_Kit/) folder  
2. Open `README.ipynb` (the main notebook)
3. Run all cells in order to explore the baseline pipeline

The notebook is organized into **5 main sections**:

1. **Section 0**: Imports and configuration
2. **Section 1**: Data loading and exploration
3. **Section 2**: Data visualization
4. **Section 3**: Baseline model training
5. **Section 4**: Model evaluation and scoring
6. **Section 5**: Submission preparation

**⚠️ Important note**: 
- This notebook provides a **simple baseline** that works but is not optimized
- **Your goal**: Improve this baseline to achieve better performance
- **Improvement suggestions**: See the [Model Architecture](#model-architecture) section above

**💡 Tips for getting started**:
1. Read and execute each section in order
2. Understand what each part does before modifying it
3. Experiment progressively (one modification at a time)
4. Visualize your results to better understand

### 🔧 Installation & Setup

#### Running the Notebook

You can run this notebook in two ways:

1. **Google Colab** (Recommended for beginners)
   - Click on the Colab badge in the notebook to open it in Google Colab
   - No local installation needed
   - Free GPU access available
   - **Note**: Update the Colab link with your actual repository URL

2. **Local Environment**
   - Requires Python 3.7+
   - Install required packages (see below)
   - Clone the repository and navigate to `Starting_Kit/` folder

#### Required Packages

The following packages are needed to run this notebook:

**Core packages**:
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `scikit-learn`: Machine learning utilities
- `tensorflow` or `keras`: Deep learning framework

Install them using:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Or in the notebook:
```python
!pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## Repository Structure

```
grain-1-generalization-ai-challenge/
├── Starting_Kit/
│   ├── README.ipynb          # Main starting kit notebook
│   ├── functions.py           # Helper functions and classes
│   └── README.md             # Additional documentation
├── Competition_Bundle/        # Files for Codabench deployment
│   ├── competition.yaml       # Competition configuration
│   ├── ingestion_program/     # Code to run participant submissions
│   ├── scoring_program/       # Code to evaluate submissions
│   ├── input_data/            # Training and test data
│   ├── reference_data/        # Test labels (ground truth)
│   └── pages/                 # Competition web pages
├── Data/                      # Original data (not in repo)
│   ├── Grain-Data-RGB/        # RGB images dataset
│   └── Grain-Data/            # Hyperspectral images dataset
└── README.md                  # This file
```

---

## Evaluation Platform

This challenge will be deployed and evaluated using **Codabench**.  
The starting kit notebook is designed to be compatible with Codabench submissions.

### Submission Types

1. **Code Submission**: Submit your trained model
   - The platform will load your model and run it on test data
   - Requires saving the model architecture and weights

2. **Result Submission**: Submit predictions directly
   - Pre-compute predictions on test data
   - Submit the predictions file

Check the competition rules to determine which type is required.

### Scoring

The scoring system computes **3 key metrics**:
- **Accuracy** (primary metric for leaderboard)
- **F1-Score (Macro)**
- **Cohen's Kappa**

The leaderboard displays **Accuracy** as the main ranking metric. All three metrics are computed and saved in `scores.json` for detailed analysis.

---

## Credits : Team & Contact


### Contact Information

- **Challenge Leader**: [Lubin LONGUEPEE] - [lubin.longuepee@gmail.com]
- **Challenge Team**: 
   - Lounès KEBDI
   - Bill TANG
   - Saykat NAIM HASSAN
   - Eloi BEURTHERET - [eloi.beurtheret02@gmail.com]
- **GitHub Repository**: https://github.com/md-naim-hassan-saykat/grain-1-generalization-ai-challenge
- **Course**: Creation of an AI Challenge
- **Institution**: Université Paris-Saclay
- **Year**: 2025-26

### Acknowledgments

- Data provided for educational purposes within the M1 AI Challenge Creation class
- Special thanks to [any acknowledgments]

### License

This starting kit and challenge materials are provided for educational purposes as part of the AI Challenge Creation course at Université Paris-Saclay.

---

## 📚 Additional Resources

### Getting Help

- Check the [GitHub Issues](https://github.com/md-naim-hassan-saykat/grain-1-generalization-ai-challenge/issues) for common questions
- Review the main repository [README.md](../README.md) for project overview
- Contact the team for specific questions

### Useful Links

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [NumPy Documentation](https://numpy.org/doc/)

---

**Good luck with the challenge! 🚀**
