# =========================================================
# TELECOM CUSTOMER CHURN PREDICTION
# STREAMLIT APPLICATION
# File: app/streamlit_app.py
# =========================================================

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}

.sub-title {
    font-size:18px;
    color:gray;
}

.metric-card {
    background-color:#f5f5f5;
    padding:20px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median()
    )

    return df

# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = load_data()

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_columns:

        df[col] = le.fit_transform(df[col])

    # Features and Target
    X = df.drop('Churn', axis=1)

    y = df['Churn']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    roc_score = roc_auc_score(y_test, y_prob)

    return (
        model,
        accuracy,
        roc_score,
        X_test,
        y_test,
        y_pred,
        y_prob
    )

# =========================================================
# LOAD EVERYTHING
# =========================================================

(
    model,
    accuracy,
    roc_score,
    X_test,
    y_test,
    y_pred,
    y_prob
) = train_model()

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📡 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Predict Customer",
        "EDA",
        "Model Performance"
    ]
)

# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    '<p class="main-title">📡 Telecom Customer Churn Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Machine Learning based Customer Churn Analysis System</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# EDA PAGE
# =========================================================

if page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    # Churn Distribution
    st.subheader("Customer Churn Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        x='Churn',
        data=df,
        palette='Set2'
    )

    st.pyplot(fig)

    # Contract vs Churn
    st.subheader("Contract Type vs Churn")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        x='Contract',
        hue='Churn',
        data=df,
        palette='Set1'
    )

    st.pyplot(fig)

    # Monthly Charges
    st.subheader("Monthly Charges Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['MonthlyCharges'],
        kde=True,
        color='blue'
    )

    st.pyplot(fig)

    # Tenure vs Churn
    st.subheader("Tenure vs Churn")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        x='Churn',
        y='tenure',
        data=df,
        palette='Set3'
    )

    st.pyplot(fig)

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:

        st.metric(
            "ROC AUC Score",
            f"{roc_score:.2f}"
        )

    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")

    fpr, tpr, threshold = roc_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(7,5))

    plt.plot(
        fpr,
        tpr,
        label='Random Forest'
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle='--'
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "Predict Customer":

    st.header("🔮 Predict Customer Churn")

    st.write(
        "Enter customer details below:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure",
            0,
            72,
            12
        )

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col3:

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.slider(
            "Monthly Charges",
            0.0,
            200.0,
            70.0
        )

        total_charges = st.slider(
            "Total Charges",
            0.0,
            10000.0,
            1000.0
        )

    # =====================================================
    # ENCODE INPUTS
    # =====================================================

    input_data = pd.DataFrame({

        'gender': [1 if gender == "Male" else 0],

        'SeniorCitizen': [senior],

        'Partner': [1 if partner == "Yes" else 0],

        'Dependents': [1 if dependents == "Yes" else 0],

        'tenure': [tenure],

        'PhoneService': [1 if phone_service == "Yes" else 0],

        'MultipleLines': [
            0 if multiple_lines == "No"
            else 1 if multiple_lines == "Yes"
            else 2
        ],

        'InternetService': [
            0 if internet_service == "DSL"
            else 1 if internet_service == "Fiber optic"
            else 2
        ],

        'OnlineSecurity': [
            0 if online_security == "No"
            else 1 if online_security == "Yes"
            else 2
        ],

        'OnlineBackup': [0],

        'DeviceProtection': [0],

        'TechSupport': [
            0 if tech_support == "No"
            else 1 if tech_support == "Yes"
            else 2
        ],

        'StreamingTV': [0],

        'StreamingMovies': [0],

        'Contract': [
            0 if contract == "Month-to-month"
            else 1 if contract == "One year"
            else 2
        ],

        'PaperlessBilling': [
            1 if paperless == "Yes" else 0
        ],

        'PaymentMethod': [
            0 if payment_method == "Electronic check"
            else 1 if payment_method == "Mailed check"
            else 2 if payment_method == "Bank transfer (automatic)"
            else 3
        ],

        'MonthlyCharges': [monthly_charges],

        'TotalCharges': [total_charges]
    })

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button("Predict Churn"):

        prediction = model.predict(
            input_data
        )

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.markdown("---")

        if prediction[0] == 1:

            st.error(
                f"⚠️ Customer is likely to churn\n\n"
                f"Churn Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ Customer is likely to stay\n\n"
                f"Churn Probability: {probability:.2%}"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>

Telecom Customer Churn Prediction System  
Built using Streamlit and Machine Learning

</center>
""", unsafe_allow_html=True)