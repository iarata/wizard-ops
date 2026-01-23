<div align="center">
  <img src="wizops.png" alt="WizOps Logo" width="200"/>
  
  # 🍽️ Nutrition Predictor
  
  *Predict nutritional values from food images using deep learning*
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📋 Overview

The goal of this project is to perform MLOps from training to deploying and serving.

## 📊 Dataset

The model is trained on a subset of the
[Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file)
containing 5 images per angle.

After processing, our dataset contains 20 images for every 5000 different
dishes and metadata for each dish corresponding to 10GB data.
The images are JPEGs, and the metadata is a CSV containing the total calories, mass,
fat, carbs, and protein associated with each dish ID.

## 🏗️ Architecture

We use **ResNet18** as the backbone, adapted for regression by replacing the
final classification layer with a small feed-forward network that outputs the nutritional
values. The pretrained backbone serves as a feature extractor, with its weights
frozen during training while the regression head is learned.

This architecture follows approaches commonly seen in Kaggle notebooks for the Nutrition5k
dataset. While published work has employed larger models like InceptionV2 and
ResNet50/101 ([Thames et al., 2021](https://arxiv.org/abs/2103.03375)), ResNet18
is a reasonable choice for purposes not focused on prediction accuracy.
