# Grain Variety Classification Challenge

## Introduction
***

Welcome to the **Grain Variety Classification Challenge**! This challenge is part of the "Creation of an AI Challenge" course at Université Paris-Saclay (2025-26).

### Objective

The goal of this challenge is to **develop an artificial intelligence model capable of automatically classifying wheat grain varieties** from images. This is a **multi-class classification problem** where you need to predict the variety number of each grain from its image.

### Why This Challenge Matters

- **Agricultural Applications**: Rapid and accurate identification of grain varieties is crucial for modern agriculture
- **Quality Assurance**: Ensures quality, traceability, and compliance of harvests
- **Automation**: Significantly improves efficiency of agricultural processes
- **Generalization Focus**: The challenge emphasizes model generalization - your model must work well on grains from different conditions (microplots, dates, imaging setups)

### Scientific Context

The data comes from grain images captured using two types of imaging techniques:
- **RGB Imaging**: Standard color images (3 channels: Red, Green, Blue)
- **Hyperspectral Imaging**: Images containing many spectral bands (more information than the human eye can perceive)

Each grain was photographed individually, and each image is associated with a known variety (the "label" or "ground truth").

## Competition Tasks
***

### Task Description

**Classification Task**: Given an image of a grain, predict its variety number.

- **Input**: Grain images (RGB or hyperspectral)
- **Output**: Variety number (integer from 2 to 9, representing 8 different varieties)
- **Type**: Multi-class classification
- **Evaluation Metric**: Accuracy (percentage of correctly classified grains)

### Key Challenge: Generalization

The challenge emphasizes **generalization** - your model must correctly classify grains from:
- **Different microplots** (32 different microplots with varying growing conditions)
- **Different acquisition dates** (2020-2021, with varying lighting and environmental conditions)
- **Different imaging conditions** (varying camera parameters, lighting, etc.)

A model that only works on training data but fails on new conditions is not useful in practice!

### Datasets

You have access to two datasets:
1. **Grain-Data-RGB**: RGB images (3 channels) - faster to process, good for starting
2. **Grain-Data**: Full hyperspectral data (multiple channels) - more information, potentially better performance

## Competition Phases
***

### Phase 1: Development Phase (February 1 - March 15, 2026)
- **Purpose**: Development and testing
- **Data**: Training data and validation data available
- **Submissions**: Unlimited submissions for testing
- **Leaderboard**: Public leaderboard for development phase

### Phase 2: Final Evaluation Phase (March 15 - March 31, 2026)
- **Purpose**: Final evaluation on test set
- **Data**: Test data (without labels) available
- **Submissions**: Limited submissions (check submission rules)
- **Leaderboard**: Final leaderboard determines winners

## How to Join This Competition?
***

1. **Login or Create Account** on [Codabench](https://www.codabench.org/)
2. **Go to the `Starting Kit` tab**
   - Download the starting kit notebook
   - Review the baseline model and data loading code
   - Understand the data structure and evaluation metrics
3. **Download the `Dummy Sample Submission`**
   - This is a sample submission file to test the submission process
4. **Go to the `My Submissions` tab**
5. **Register in the Competition**
6. **Submit your solution**
   - For code submission: Submit your trained model
   - Follow the format specified in the starting kit

## Submissions
***

This competition allows **code submissions**. Participants must submit:
- Their trained model (architecture + weights)
- Any necessary preprocessing code
- Model loading and prediction code

The platform will:
1. Load your model
2. Run it on the test data
3. Evaluate the predictions automatically

**Submission Format**: See the starting kit for detailed instructions on how to prepare your submission.

## Evaluation

### Metric: Accuracy

**Accuracy** is the primary evaluation metric:
- **Definition**: Percentage of correctly classified grains
- **Formula**: `Accuracy = (Number of correct predictions) / (Total number of predictions)`
- **Range**: 0.0 to 1.0 (or 0% to 100%)

**Why Accuracy?**
- All varieties are equally important
- Easy to interpret and compare
- Directly answers "how many grains are correctly identified"

Additional metrics (Precision, Recall, F1-score) are computed per class for detailed analysis.

## Timeline
***

- **Challenge Launch**: February 1, 2026
- **Development Phase**: February 1 - March 15, 2026
- **Final Evaluation Phase**: March 15 - March 31, 2026
- **Results Announcement**: Early April 2026

## Prizes & Recognition
***

- Top performers will be recognized
- Best generalization performance award
- Most innovative approach award

*Note: This is an educational challenge. Prizes may vary.*

## Credits
***

**Group 1 - Grain (Generalization)**
- Team members from Université Paris-Saclay
- Course: Creation of an AI Challenge (2025-26)

**Data Source**
- Grain imaging data provided for educational purposes
- Data collected from multiple microplots over 2020-2021

**Acknowledgments**
- Université Paris-Saclay for hosting this challenge
- Course instructors and organizers

## Contact
***

- **GitHub Repository**: [grain-1-generalization-ai-challenge](https://github.com/md-naim-hassan-saykat/grain-1-generalization-ai-challenge)
- **Starting Kit**: Available in the repository `Starting_Kit/` folder
- **Issues**: Report issues or ask questions via GitHub Issues

For specific questions about the challenge, please contact the challenge organizers through the Codabench platform.

---

**Good luck and happy coding! 🚀**