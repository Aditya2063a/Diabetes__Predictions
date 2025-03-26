# Diabetes Prediction Using Machine Learning

This repository contains code to analyze and predict diabetes outcomes using various machine learning techniques including K-Nearest Neighbors, Decision Trees, and Deep Learning (MLP). The code loads a diabetes dataset, performs exploratory data analysis (EDA), builds different models, and visualizes important aspects such as feature importances and weight matrices.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
  - [Data Loading and EDA](#data-loading-and-eda)
  - [K-Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Feature Importance Visualization](#feature-importance-visualization)
  - [Deep Learning with MLP](#deep-learning-with-mlp)

    
## Overview

This project demonstrates how to:

- Load and explore a diabetes dataset.
- Visualize the distribution of diabetes outcomes.
- Split the dataset into training and testing sets.
- Build and evaluate different classification models:
  - **K-Nearest Neighbors (KNN)**
  - **Decision Tree Classifier**
  - **Deep Learning using Multi-Layer Perceptron (MLP)**
- Visualize feature importances in decision trees and weight matrices in neural networks.

## Dataset

The dataset (`diabetes.csv`) contains the following columns:

- **Pregnancies**
- **Glucose**
- **BloodPressure**
- **SkinThickness**
- **Insulin**
- **BMI**
- **DiabetesPedigreeFunction**
- **Age**
- **Outcome**

The target variable is `Outcome` where:

- `0` indicates the absence of diabetes.
- `1` indicates the presence of diabetes.

## Dependencies

Ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

