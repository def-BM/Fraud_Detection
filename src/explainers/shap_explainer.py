import shap
import joblib
import pandas as pd
from pathlib import Path
from utils.download_model import download_files

# Download required files
download_files()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models/xgb_fraud_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models/feature_columns.pkl"

# Load artifacts
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

def generate_explaination(scaled_df: pd.DataFrame, top_k=5):
    # Generate human-readable fraud explanation using SHAP values

    # SHAP Values
    shap_values = explainer.shap_values(scaled_df)

    # Convert to dataframe
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    # Get top contributing features
    top_features = shap_df.reindex(
        shap_df.shap_value.abs().sort_values(ascending=False).index
    ).head(top_k)

    # Generate explanation text
    explanations = []
    for _, row in top_features.iterrows():
        impact = "increased" if row.shap_value > 0 else "decreased"
        explanations.append(
            f"{row.feature} {impact} the fraud risk significantly"
        )
    
    return{
        "top_features": top_features.to_dict(orient="records"),
        "explanation_text": explanations
    }
