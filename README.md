# Customer-Churn-Prediction-for-Telecom-Companies

A Machine Learning project that predicts whether a telecom customer is likely to churn (leave the service) using customer demographics, account information, and service usage patterns.

---

# Project Information

## Project Title
Customer Churn Prediction for Telecom Companies

## Team Members
- Anamika A (Msc Computer Science and Data Analytics)
- Ashna Jabin NK (Msc Data analytics and Computational Science)
- Aparna Sreenivasan ( Msc Bio AI)


# Problem Statement

Customer churn is one of the major challenges faced by telecom companies. Losing existing customers directly impacts revenue and business growth. Telecom providers need a reliable system to identify customers who are likely to leave the service so that retention strategies can be implemented in advance.

This project aims to build a machine learning-based churn prediction system capable of identifying potential churn customers using customer behavior and service-related data.

---

# Motivation

Customer retention is more cost-effective than acquiring new customers. By predicting churn early, telecom companies can:

- Improve customer satisfaction
- Reduce revenue loss
- Design targeted retention campaigns
- Understand customer behavior patterns
- Improve business decision-making

---

# Dataset Description

This project uses the IBM Telco Customer Churn Dataset.

## Dataset Details

| Attribute | Value |
|---|---|
| Dataset Source | IBM Telco Customer Churn Dataset |
| Total Records | 7,043 Customers |
| Total Features | 21 Columns |
| Target Variable | Churn |

## Important Features

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges

## Class Distribution

- Customers who stayed: Majority class
- Customers who churned: Minority class

Since the dataset is imbalanced, SMOTE (Synthetic Minority Oversampling Technique) was applied during preprocessing.

---

# Project Methodology

The project follows a complete Machine Learning lifecycle.

## 1. Data Understanding
- Loaded and inspected the dataset
- Identified data types and missing values
- Understood customer-related attributes

## 2. Exploratory Data Analysis (EDA)

Performed visual and statistical analysis to identify churn patterns.

### Key Insights
- Customers with month-to-month contracts have higher churn rates.
- Customers with higher monthly charges are more likely to churn.
- Customers with short tenure tend to leave the service earlier.
- Customers using Fiber Optic Internet show higher churn.
- Customers paying via Electronic Check exhibit higher churn.
- Senior citizens have slightly higher churn rates.

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Handled missing values
- Converted categorical variables into numerical form
- Feature scaling using StandardScaler
- Train-test split
- Applied SMOTE for class balancing

---

## 4. Model Building

The following machine learning models were trained and evaluated:

### Logistic Regression
A simple and interpretable baseline classification model.

### Random Forest Classifier
An ensemble learning method using multiple decision trees.

### XGBoost Classifier
A gradient boosting algorithm with strong predictive capability.

---

# Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 75.09% |
| Random Forest Classifier | 77.29% |
| XGBoost Classifier | 77.22% |

## Best Performing Model
✅ Random Forest Classifier

---

# Evaluation Metrics

## ROC-AUC Score
- XGBoost ROC-AUC Score: 0.721

## Other Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve

---

# Screenshots

## Application Dashboard
<img width="1917" height="802" alt="image" src="https://github.com/user-attachments/assets/749f847f-a813-40f9-a49c-7e47ffc0670a" />

## ROC Curve / Model Results
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/df24bd86-e244-4898-a8c1-028f68188c5c" />

## EDA Visualizations
<img width="699" height="473" alt="image" src="https://github.com/user-attachments/assets/4d461a72-1c55-4f4a-b8d3-a2b0d5c27945" />

<img width="707" height="473" alt="image" src="https://github.com/user-attachments/assets/cee79f19-df79-46e7-a396-cb4d0c17757f" />



# Streamlit Deployment

## Live Application

Streamlit App:  
https://customer-churn-prediction-for-telecom-companies.streamlit.app/

---

# Project Structure

```bash
Customer-Churn-Prediction-for-Telecom-Companies/
│
├── data/
│   └── telecom_churn.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 03_eda.ipynb
│   └── 05_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── random_forest_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

# Installation and Setup

## Clone the Repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction-for-Telecom-Companies.git
```

## Navigate to Project Directory

```bash
cd Customer-Churn-Prediction-for-Telecom-Companies
```

## Install Required Libraries

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- SMOTE

---

# Future Improvements

- Hyperparameter tuning
- Deep learning implementation
- Real-time prediction system
- Deployment using Docker and Cloud platforms
- Advanced customer segmentation

---

# Conclusion

This project successfully predicts telecom customer churn using machine learning techniques. Among the evaluated models, the Random Forest Classifier achieved the best performance. The project demonstrates how data analytics and machine learning can help businesses reduce customer attrition and improve retention strategies.

---

# References

- IBM Telco Customer Churn Dataset
- Scikit-learn Documentation
- XGBoost Documentation
- Streamlit Documentation
