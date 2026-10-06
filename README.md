# Detecting Heart Abnormality using ECG with CART

## Problem Statement

The objective of this project is to develop a machine learning model for detecting and classifying heart abnormalities using ECG-related features. A CART (Classification and Regression Tree) decision tree is used as the primary classification model, with Random Forest used as a comparison model.

## Dataset

The project uses the UCI Arrhythmia Dataset.

- 452 samples
- 279 original attributes
- ECG-related features
- 16 possible class labels in the dataset
- Missing values represented by `?`
- Significant class imbalance

## Methodology

The following steps were performed:

1. Load the UCI Arrhythmia Dataset.
2. Handle missing values.
3. Remove the specified feature.
4. Separate input features and target labels.
5. Perform median imputation for missing numerical values.
6. Split the dataset into training and testing sets using an 80:20 split.
7. Train a CART Decision Tree using Gini impurity.
8. Perform hyperparameter tuning using GridSearchCV.
9. Evaluate the CART model.
10. Compare CART performance with Random Forest.
11. Analyze feature importance.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
ML-Mini-Project/
│
├── data/
│   └── arrhythmia.data
│
├── results/
│   ├── cart_results.csv
│   └── feature_importance.csv
│
├── ECG_CART_Model.ipynb
│
└── README.md
