# Customer-Churn-Prediction-for-Telecom-Companies
A machine learning project to predict whether a telecom customer will churn (leave the service) using customer demographic information, account details, and service usage data.

# Project Overview
Customer churn prediction is a common business problem in the telecom industry.

This project:

1. Performs exploratory data analysis (EDA)
2. Cleans and preprocesses the data
3. Handles class imbalance using SMOTE
4. Trains multiple machine learning models
5. Compares model performance and identifies the best model 
6. Identifies important churn factors

# Dataset

This project uses the popular IBM Telco Customer Churn dataset.
Number of rows: 7,043 customers
Number of columns: 21 features

Target Variable - Churn

# Exploratory Data Analysis (EDA)
EDA was used to understand customer behavior and identify factors associated with churn.
Customers on month-to-month contracts, those with higher monthly charges, and those with shorter tenure are more likely to leave. Those using fiber optic internet and those paying by electronic check also exhibit higher churn rates.Senior citizens show slightly higher churn compared to other customers. 

# Data Preprocessing

# Models used:
1. Logistic Regression
A simple and interpretable baseline model.

2. Random Forest Classifier
An ensemble tree-based model.

3. XGBoost Classifier
A gradient boosting model with strong predictive performance.

# Model performance
Model	Accuracy
1. Logistic Regression:	75.09%
2. Random Forest:	77.29%
3. XGBoost:	77.22%
The best model in Random Forest

# Classification Report

# ROC-AUC Score
XGBoost ROC-AUC: 0.721

# Deployment
deployed app : https://customer-churn-prediction-for-telecom-companies.streamlit.app/

<img width="1917" height="802" alt="image" src="https://github.com/user-attachments/assets/749f847f-a813-40f9-a49c-7e47ffc0670a" />

