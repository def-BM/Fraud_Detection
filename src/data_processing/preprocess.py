import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = 'data/raw_data/creditcard.csv'
PROCESSED_DATA_DIR = 'data/processed_data/'
MODEL_DIR = 'models/'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Data
def load_data(path):
    df = pd.read_csv(path)
    print(f"Data loaded with shape: {df.shape}")
    return df

# Preprocess Pipeline
def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()

    feature_columns = X.columns.tolist()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
    )

    fraud_samples = X_train[y_train == 1].sample(50, random_state=42)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, fraud_samples

# Save Artifacts
def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_columns, fraud_samples):
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    joblib.dump(fraud_samples, "models/fraud_prototypes.pkl")

    print("Artifacts saved successfully.")

# Main Execution    
if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_columns, fraud_samples = preprocess_data(df)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_columns, fraud_samples)

    print("Data preprocessing completed.")