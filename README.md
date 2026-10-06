# Home Loan Approval Prediction

## Project Overview

This project develops a machine learning classification workflow to predict **historical home loan approval status** using applicant financial, demographic, credit, and property-related information.

The project covers data exploration, data cleaning, exploratory data analysis, preprocessing, model training, hyperparameter tuning, validation-based model comparison, final evaluation on an untouched test set, and prediction on an external test dataset.

> **Note:** The target represents historical approval decisions, not borrower default or repayment risk.

## Dataset

The training dataset contains **614 observations** before preprocessing.

The target variable is:

- `Loan_Status = 1` → Approved
- `Loan_Status = 0` → Rejected

Main features include:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

An additional external test dataset containing **367 applications** was used to generate final predictions.

## Project Workflow

1. Data loading and inspection
2. Missing-value and duplicate analysis
3. Exploratory Data Analysis (EDA)
4. Outlier and skewness handling
5. Stratified train/validation/test splitting
6. Numerical and categorical preprocessing
7. Model training
8. Hyperparameter tuning using stratified cross-validation
9. Validation-based model comparison
10. Final model selection
11. Evaluation on the untouched test set
12. Model saving
13. External test predictions

## Data Preprocessing

The preprocessing workflow included:

- Removing `Loan_ID` from the model features
- Mapping the target variable from `Y/N` to `1/0`
- Treating `Credit_History` as a categorical feature
- Removing observations with `CoapplicantIncome >= 10000` from the training data as part of the experimental preprocessing strategy
- Applying `log1p` transformation to `ApplicantIncome` and `LoanAmount`
- Median imputation for numerical missing values
- Most-frequent imputation for categorical missing values
- Standardization of numerical features
- One-hot encoding of categorical features

The data was divided using stratified sampling into:

- **Training:** 388 observations
- **Validation:** 98 observations
- **Test:** 122 observations

## Models Evaluated

Five classification algorithms were evaluated:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

Hyperparameter tuning was performed using **GridSearchCV** with **5-fold Stratified Cross-Validation**.

## Validation Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.8163 | 0.7976 | 0.9853 | 0.8816 | **0.7588** |
| SVC | 0.8163 | 0.7976 | 0.9853 | 0.8816 | 0.7505 |
| Random Forest | 0.8061 | 0.8182 | 0.9265 | 0.8690 | 0.7397 |
| Logistic Regression | 0.8265 | 0.8072 | 0.9853 | 0.8874 | 0.7309 |
| Decision Tree | 0.8265 | 0.8072 | 0.9853 | 0.8874 | 0.7047 |

ROC-AUC was used as the primary model-selection metric.

## Final Model

The tuned **K-Nearest Neighbors (KNN)** classifier achieved the highest validation ROC-AUC and was selected as the final model.

Best hyperparameters:

```python
KNeighborsClassifier(
    n_neighbors=15,
    weights="distance",
    p=1
)
```

After model selection, the final model was retrained using the combined training and validation data.

## Final Test Results

The selected KNN model was evaluated once on the previously untouched test set.

| Metric | Result |
|---|---:|
| Accuracy | **0.72** |
| ROC-AUC | **0.7249** |
| Weighted F1-score | **0.68** |

### Class-Specific Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Rejected (0) | 0.62 | 0.26 | 0.37 |
| Approved (1) | 0.74 | 0.93 | 0.82 |

The model demonstrated strong sensitivity to historically approved applications, but substantially weaker recall for historically rejected applications.

This indicates a tendency toward predicting the majority approval class.

## External Test Predictions

The final model was also applied to the external test dataset containing **367 applications**.

Since the external dataset does not contain `Loan_Status`, it was used only for generating predictions rather than calculating evaluation metrics.

The generated output contains:

- `Loan_ID`
- `Predicted_Loan_Status`
- `Approval_Probability`

The predictions are saved in:

`loan_approval_predictions.csv`

## Saved Model

The trained model is saved as:

`home_loan_approval_knn_pipeline.pkl`

The saved Scikit-learn pipeline contains:

- Missing-value imputation
- Numerical scaling
- Categorical one-hot encoding
- KNN classifier

The `Credit_History` mapping and `log1p` transformations for `ApplicantIncome` and `LoanAmount` were performed before the Scikit-learn pipeline. Therefore, the same transformations must be applied to new raw observations before prediction.

## Project Structure

```text
Home-Loan-Approval-Prediction/
│
├── home-loan-approval-prediction.ipynb
├── loan_sanction_train.csv
├── loan_sanction_test.csv
├── loan_approval_predictions.csv
├── home_loan_approval_knn_pipeline.pkl
└── README.md
```

## Limitations

- The dataset is relatively small.
- The target classes are imbalanced, with approved applications representing the majority class.
- The final model has low recall for rejected applications.
- The dataset represents historical approval decisions and should not be interpreted as a measure of borrower default risk.
- Some preprocessing decisions, including outlier handling and log transformations, were performed before the train/validation/test split as part of this experimental workflow.
- Sensitive demographic variables such as gender should be considered carefully in any real-world lending application.

## Future Improvements

Future versions of the project could explore:

- Class-imbalance handling techniques
- Decision-threshold optimization using training/validation data
- Additional feature engineering
- Larger and more representative datasets
- Probability calibration
- Fairness assessment across demographic groups
- A fully end-to-end preprocessing pipeline for raw observations

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Conclusion

Among the evaluated classifiers, tuned KNN achieved the highest validation ROC-AUC of **0.7588** and was therefore selected as the final model.

On the untouched test set, the model achieved an accuracy of **0.72** and ROC-AUC of **0.7249**. It performed particularly well at identifying historically approved applications, with a recall of **0.93**, but showed weaker performance for rejected applications.

Overall, the project demonstrates a complete machine learning classification workflow while also highlighting the importance of evaluating class-specific performance and documenting model limitations.