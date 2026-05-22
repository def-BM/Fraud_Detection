from utils.download_model import download_files
from pathlib import Path
import joblib
import pandas as pd

download_files()
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models/xgb_fraud_model.pkl"
# SCALER_PATH = BASE_DIR / "models/scaler.pkl"

# Config
FRAUD_THRESHOLD = 0.7 # Business decision threshold

# Load Artifacts
model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# Fraud Decisin Function
def fraud_decision(scaled_df: pd.DataFrame):
    """
    Input: scaled_df -> shape (n_features,)
    Output:
        decision -> ALLOW / BLOCK
        probability -> fraud probability
    """


    # predict probability
    fraud_prob = model.predict_proba(scaled_df)[0][1]

    # Decision logic
    decision = "BLOCK" if fraud_prob >= FRAUD_THRESHOLD else "ALLOW"

    return {
        'decision': decision,
        'fraud_probability': round(float(fraud_prob), 4)
    }
