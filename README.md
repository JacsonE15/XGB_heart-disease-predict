# Heart Disease Prediction (UCI Dataset)

## 1. Project Description
This project builds a heart disease prediction model using the UCI Heart Disease dataset.  
The pipeline includes exploratory data analysis (EDA), data preprocessing, baseline Logistic Regression, XGBoost modeling, hyperparameter tuning, and SHAP-based model interpretation.

## 2. Dataset
- Source: UCI Heart Disease dataset (Cleveland + Hungary + Switzerland + VA Long Beach)
- Target: `target` (0 = no heart disease, 1 = heart disease)
- Main features: age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), ST depression (oldpeak), exercise-induced angina (exang), thal, ca, etc.

## 3. Methods
- EDA: distribution plots, missing values, skewness, boxplots vs. target.
- Preprocessing:
  - Median imputation for continuous variables.
  - Mode imputation for categorical variables.
  - Outlier clipping for `chol`, `trestbps`, `oldpeak`.
  - Log transform for `oldpeak`.
  - Label encoding for categorical features.
- Models:
  - Logistic Regression (baseline).
  - XGBoost classifier:
    - Baseline model.
    - RandomizedSearchCV with 5-fold cross-validation for hyperparameter tuning.
- Explainability:
  - Feature importance.
  - SHAP summary plot and force plots for local explanations.

## 4. Results (Test Set)
- Logistic Regression:
  - ROC-AUC ≈ 0.895.
- XGBoost (baseline):
  - ROC-AUC ≈ 0.905.
- XGBoost (tuned):
  - Accuracy ≈ 0.84–0.86.
  - ROC-AUC ≈ 0.93.
  - Recall for positive class (heart disease) ≈ 0.90.
SHAP analysis highlights `cp`, `oldpeak`, `exang`, `chol`, `age`, and `sex` as important predictors.

## 5. Files
- `part1.ipynb` – EDA and preprocessing.
- `part2_xgboost.ipynb` – Modeling, tuning, evaluation, and SHAP analysis.
- `data/heart_disease_uci.csv` – Processed dataset (if included).
- `plots/` – Saved figures for feature importance and SHAP (optional).

## 6. How to Run
1. Create a Python environment (e.g., conda) and install required packages (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`).
2. Open the notebook(s) in Jupyter or VS Code.
3. Run all cells in order.
