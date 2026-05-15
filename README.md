# 📡 Telecom Customer Churn Prediction - Streamlit Application

A Machine Learning powered Streamlit web application for predicting telecom customer churn using customer demographic, service, and billing information.

---

# 🚀 Project Overview

This Streamlit application helps predict whether a telecom customer is likely to churn (leave the service) based on customer details and subscription behavior.

The application includes:
- 🔮 Real-time churn prediction
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Model performance visualization
- 📈 ROC Curve and Confusion Matrix
- 📋 Classification Report

---

# 🧠 Machine Learning Model

The application uses:

- **Random Forest Classifier**
- **Scikit-learn**
- **Label Encoding**
- **Train-Test Split**
- **ROC-AUC Evaluation**

---

# 📂 Features of the Application

## 🔮 Predict Customer Churn
Users can:
- Enter customer details
- Select service information
- Input billing details
- Predict churn probability instantly

---

## 📈 Exploratory Data Analysis (EDA)

The application provides:
- Customer churn distribution
- Contract type vs churn analysis
- Monthly charges distribution
- Tenure vs churn analysis

---

## 🤖 Model Performance Dashboard

Displays:
- Accuracy Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Classification Report

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

# 📂 Project Structure

```bash
Telecom-Churn-Prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 03_eda.ipynb
│   └── 05_model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
