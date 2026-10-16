# Detecting Heart Abnormality Using ECG with CART

**Machine Learning Mini-Project - Team 31**

## Overview

This project implements a multiclass Classification and Regression Tree (CART) model to classify records from the UCI Arrhythmia dataset into their original arrhythmia class labels. It compares a baseline CART classifier with a hyperparameter-tuned CART classifier.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Records:** 452
- **Original input attributes:** 279
- **Target:** Original arrhythmia class label
- **Observed target labels in the supplied file:** 1–10, 14, 15, and 16

## Preprocessing

1. Read the dataset with `?` treated as a missing value.
2. Convert input attributes to numeric values.
3. Remove Attribute 14 (zero-based column index 13), which has 376 missing values out of 452 records (83.2%).
4. Retain 278 input features.
5. Impute remaining numerical missing values with the median using a scikit-learn `Pipeline`.
6. Split data into 70% training and 30% testing using stratified sampling (`random_state=42`).


## Models

### Baseline multiclass CART
- `DecisionTreeClassifier`
- Gini impurity
- Default tree complexity settings
- `random_state=42`

### Tuned multiclass CART
Hyperparameters are selected using `GridSearchCV`, 3-fold `StratifiedKFold`, and macro-F1 scoring. The search covers:
- `max_depth`: `None`, 5, 7, 10, 12, 15
- `min_samples_leaf`: 1, 2, 3, 5
- `class_weight`: `None`, `balanced`
- `ccp_alpha`: 0, 0.001, 0.005, 0.01

Best parameters in the recorded notebook run:
```python
{
    "max_depth": 12,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "ccp_alpha": 0.01
}
```

## Results

Results recorded in the notebook for the same held-out test set (136 records):

| Metric | Baseline multiclass CART | Tuned multiclass CART |
|---|---:|---:|
| Accuracy | 69.12% | 55.88% |
| Balanced accuracy | 36.98% | 39.96% |
| Macro F1-score | 0.3657 | 0.3666 |

The baseline is retained as the primary model because it has higher overall accuracy. The tuned model has slightly higher balanced accuracy and macro F1, but lower overall accuracy. These results are limited by the small dataset and severe class imbalance; rare classes have very few examples, so their estimates are unstable.

## Requirements

Suggested environment:
- Python 3.10 or later
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Clone the private repository:
   ```bash
   git clone https://github.com/nihira007/ML-Mini-Project
   cd ML-Mini-Project
   ```
2. Create and activate a virtual environment (recommended).
3. Install dependencies using the command above.
4. Open `ECG_CART_Notebook.ipynb`.
7. Run all cells from top to bottom.


## Notebook Outputs

The notebook includes:
- Dataset inspection and class distribution
- Baseline CART metrics and classification report
- Grid-search tuning and tuned CART metrics
- Side-by-side metric comparison
- Baseline confusion matrix
- Decision-tree visualization (first levels)
- Top-10 feature-importance table and chart
- A sample-row prediction demonstration

## Reproducibility Notes

- The train-test split uses `random_state=42`.
- The decision trees use `random_state=42`.
- Grid search uses 3-fold stratified cross-validation and macro-F1 scoring.
- Because some classes have only one example, scikit-learn may emit a warning during 3-fold stratified cross-validation. This is an important dataset limitation, not a guarantee that every class can be validated reliably.

## 

- Nihira — PES2UG24AM106
- Shreya — PES2UG24AM154
