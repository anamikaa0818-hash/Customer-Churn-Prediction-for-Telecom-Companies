# =========================================================
# TELECOM CUSTOMER CHURN PREDICTION
# PREPROCESSING PIPELINE
# File: src/preprocessing.py
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATASET
# =========================================================

def load_data(filepath):

    """
    Load telecom churn dataset
    """

    df = pd.read_csv(filepath)

    print("\nDataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    return df


# =========================================================
# CLEAN DATA
# =========================================================

def clean_data(df):

    """
    Perform data cleaning
    """

    # Remove customerID column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    # Fill missing values
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )

    print("\nData Cleaning Completed")

    return df


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

def encode_features(df):

    """
    Encode categorical columns
    """

    le = LabelEncoder()

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_columns:

        df[col] = le.fit_transform(df[col])

    print("\nCategorical Encoding Completed")

    return df


# =========================================================
# SPLIT FEATURES AND TARGET
# =========================================================

def split_features_target(df):

    """
    Split features and target variable
    """

    X = df.drop('Churn', axis=1)

    y = df['Churn']

    print("\nFeature Target Split Completed")

    return X, y


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

def split_train_test(X, y):

    """
    Split dataset into train and test sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Test Split Completed")

    print("X_train Shape:", X_train.shape)
    print("X_test Shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# =========================================================
# FEATURE SCALING
# =========================================================

def scale_features(X_train, X_test):

    """
    Scale numerical features
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed")

    return X_train_scaled, X_test_scaled, scaler


# =========================================================
# COMPLETE PREPROCESSING PIPELINE
# =========================================================

def preprocess_pipeline(filepath):

    """
    Complete preprocessing pipeline
    """

    # Load Data
    df = load_data(filepath)

    # Clean Data
    df = clean_data(df)

    # Encode Features
    df = encode_features(df)

    # Split Features and Target
    X, y = split_features_target(df)

    # Train Test Split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Scale Features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    print("\nPreprocessing Pipeline Completed Successfully")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    # Dataset Path
    filepath = r"data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_pipeline(filepath)

    print("\nData Ready for Model Training")