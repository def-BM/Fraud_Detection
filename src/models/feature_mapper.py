import joblib
import numpy as np
import pandas as pd

FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
fraud_prototypes = joblib.load("models/fraud_prototypes.pkl")

def map_to_model_features(behavioral_features: dict):
    """
    Maps behavioral inputs to model features using real fraud prototypes.
    """

    # ----- Injection Behavioral risk signals -----
    risk_score = (
        behavioral_features["velocity"] * 0.15 +
        behavioral_features["location_risk"] * 2 +
        behavioral_features["device_type"] * 2
    )

    # ---- Choose base pattern ------
    if risk_score >= 2.5:
        # High-risk -> real fraud pattern
        base = fraud_prototypes.sample(1).iloc[0].values
    else:
        # Low-risk -> normal transaction
        base = np.zeros(len(FEATURE_ORDER))
    
    # ---- Build DataFrame -----
    df = pd.DataFrame([base], columns=FEATURE_ORDER)

    # ------ Inject user-specific values -----
    df['Time'] = behavioral_features["txn_hour"] * 3600  # Convert hours to seconds
    df['Amount'] = behavioral_features["amount"]

    return df
