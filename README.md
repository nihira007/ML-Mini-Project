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

Results
The tuned CART model achieved an accuracy of 72.53%.
The Random Forest comparison model achieved:
- Accuracy: 73.63%
- Precision: 67.53%
- Recall: 73.63%
- F1 Score: 68.49%
The final model comparison and additional visualizations will be added to the results/ directory.
Conclusion
CART was successfully implemented for ECG-based heart abnormality classification. Hyperparameter tuning was used to improve the decision tree performance and control model complexity. Random Forest was used as a comparison model and achieved slightly higher test accuracy than the tuned CART model.
Team Members
Team 31
- Shreya Raghuraj - PES2UG24AM154
- Nihira - PES2UG24AM106
