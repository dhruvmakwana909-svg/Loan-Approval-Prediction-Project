<img width="1536" height="1024" alt="Loan Project banner" src="https://github.com/user-attachments/assets/89fe85d8-92d7-4573-ae52-3ec87e366405" />

🏦 Loan Approval Prediction – Machine Learning Project

A complete end-to-end Machine Learning pipeline to predict Loan Approval (Yes/No) using Logistic Regression and Random Forest.
The project covers data cleaning, EDA visualizations, preprocessing pipelines, model training, and classification performance evaluation.

Project Overview

This project builds an automated ML workflow for loan approval prediction:
Explore & clean the raw dataset
Handle missing values using best practices
Visualize categorical and numerical features
Build preprocessing pipelines for both types of data
Train two ML models (Logistic Regression & Random Forest)
Compare accuracy, confusion matrix, and classification report
Produce a production-ready prediction system

Key Features
1. Data Inspection & Cleaning

Displays shape, datatypes, missing values
Fills missing categorical values using mode
Fills numeric fields using median
Removes Loan_ID (non-predictive)
Encodes target variable:
Y → 1
N → 0

2. Exploratory Data Analysis (EDA)

Visualizations generated using Seaborn:
Categorical Variable Distribution
Gender
Married Status
Education
Self-employed
Property Area
Loan Status
Numeric Feature Distribution
Applicant Income
Co-applicant Income
Loan Amount
Loan Term
Credit History
These help understand patterns influencing loan approval.

3. Preprocessing Pipeline

Uses ColumnTransformer + Pipeline:
Numerical:

Imputation → Median
Scaling → StandardScaler

Categorical:

Imputation → Most Frequent
Encoding → OneHotEncoder (drop='first')
This ensures clean and ML-ready transformed data with no manual processing.

4. Model Building

Two supervised ML classification models:

Model	Purpose
Logistic Regression	Baseline classification model
Random Forest Classifier	High-accuracy ensemble method
Both models use the same preprocessing pipeline, ensuring fair comparison.

5. Model Evaluation

Reports include:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
Example output:
Accuracy: 0.82

Confusion Matrix:
[[23  4]
 [ 6 31]]

Classification Report:
              precision  recall  f1-score
              
      0          0.79     0.85     0.82

      1          0.89     0.84     0.86


Random Forest predictions also computed for comparison.

<img width="571" height="455" alt="l_1" src="https://github.com/user-attachments/assets/c259835f-28ea-4233-8e7a-98104549f3b6" />

<img width="571" height="455" alt="l_2" src="https://github.com/user-attachments/assets/770a8e03-436d-4d52-a0c0-12e5ffe364cc" />

<img width="571" height="455" alt="l-3" src="https://github.com/user-attachments/assets/5b93fae4-b9da-4a21-b6ea-1e8e3632fe78" />

<img width="571" height="455" alt="l-4" src="https://github.com/user-attachments/assets/d7e62fda-aaeb-465f-b844-752ebf29ef8f" />

<img width="571" height="455" alt="l-5" src="https://github.com/user-attachments/assets/701458c6-6095-41a7-961b-64b3d6ad9fa8" />

<img width="571" height="455" alt="l-6" src="https://github.com/user-attachments/assets/5a96937c-5b14-4d10-8deb-4ec119b07cca" />

<img width="540" height="316" alt="l-7" src="https://github.com/user-attachments/assets/248a104f-f031-4315-812a-b1d881255d20" />

<img width="540" height="316" alt="l-8" src="https://github.com/user-attachments/assets/a96ab794-7bdf-4099-b843-60184fc91f55" />

<img width="540" height="316" alt="l-9" src="https://github.com/user-attachments/assets/e725b992-7095-4109-8da1-b183abd60dc8" />

<img width="550" height="316" alt="l-10" src="https://github.com/user-attachments/assets/cbc8d89f-0158-43cc-8722-32bd422e357a" />

<img width="540" height="316" alt="l-11" src="https://github.com/user-attachments/assets/b3898511-6e38-4209-8256-41a887dd066d" />
