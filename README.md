# Diabetes Prediction Analysis

This project focuses on predicting diabetes outcomes using various machine learning techniques, including K-Nearest Neighbors (KNN), Decision Trees, and Deep Learning with a Multi-Layer Perceptron (MLP). The workflow includes data loading, exploratory data analysis (EDA), model training, hyperparameter tuning, and feature importance visualization.

---

## Table of Contents

- [Code Structure](#code-structure)
- [Data Loading and EDA](#data-loading-and-eda)
- [K-Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
- [Decision Tree Classifier](#decision-tree-classifier)
- [Feature Importance Visualization](#feature-importance-visualization)
- [Deep Learning with MLP](#deep-learning-with-mlp)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Code Structure

The project follows a structured approach, beginning with data preprocessing and visualization, followed by model training and evaluation. The major components include:

1. **Data Loading and EDA**
2. **K-Nearest Neighbors Classifier**
3. **Decision Tree Classifier**
4. **Feature Importance Visualization**
5. **Deep Learning with MLP**

---

## Data Loading and EDA

- The dataset is loaded using `pandas.read_csv()`.
- Column names and the first few rows of the dataset are displayed.
- The dataset dimensions and class distribution for the `Outcome` variable are analyzed.
- A countplot using Seaborn visualizes the distribution of diabetes outcomes.
- `DataFrame.info()` is printed to show data types and memory usage.

---

## K-Nearest Neighbors Classifier

- The dataset is split into training and testing sets using `train_test_split()`, with stratification based on `Outcome`.
- A loop iterates through `n_neighbors` values from 1 to 10 to determine the best-performing model.
- Training and test accuracies are plotted for performance comparison.
- The final KNN model is selected and evaluated using an optimal number of neighbors (e.g., 9).

---

## Decision Tree Classifier

- A Decision Tree Classifier is trained on the dataset.
- Initially, a tree with no depth constraint is used, leading to perfect training accuracy but lower test accuracy due to overfitting.
- A second tree is trained with a maximum depth of 3 to balance training and test accuracy.
- Feature importances are printed and visualized to understand their contribution to decision-making.

---

## Feature Importance Visualization

- A helper function `plot_feature_importances_diabetes()` is defined.
- This function generates a horizontal bar plot of feature importances from the trained Decision Tree model.

---

## Deep Learning with MLP

- A Multi-Layer Perceptron (MLP) classifier is implemented to predict diabetes outcomes.
- Initially, the MLP is trained on unscaled data, showing moderate performance.
- The dataset is standardized using `StandardScaler`, and the MLP is retrained.
- Two experiments are conducted:
  1. Using default parameters.
  2. Increasing the number of iterations and adjusting the regularization parameter (`alpha`) for better convergence.
- The final weight matrix from the first layer of the neural network is visualized using a heatmap.

---

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis:**
   ```bash
   python analysis.py
   ```
4. **Explore the results:**
   - EDA visualizations
   - KNN and Decision Tree performance plots
   - Feature importance bar plots
   - MLP training performance and weight heatmap

---

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- (Optional) TensorFlow/Keras for deep learning

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
