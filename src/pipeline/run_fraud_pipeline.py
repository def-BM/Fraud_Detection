import pandas as pd
import joblib
from decision_engine.fraud_decision import fraud_decision
from explainers.shap_explainer import generate_explaination
from llm.llm_explainer import genrate_llm_explanation

# Load artifacts
scaler = joblib.load('models/scaler.pkl')
FEATURE_NAMES = joblib.load('models/feature_columns.pkl')

def run_fraud_pipeline(transaction_df: pd.DataFrame):
    # ----- Validation ------
    if not isinstance(transaction_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Safety check
    if transaction_df.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Expected 30  features (V1-V28, Amount, Time)")

    transaction_df = transaction_df[FEATURE_NAMES]
    # Scale features
    scaled_df = pd.DataFrame(
        scaler.transform(transaction_df),
        columns=FEATURE_NAMES
    )

    # Step 1: ML decision
    decision_result = fraud_decision(scaled_df)
    decision = decision_result["decision"]
    fraud_probability = decision_result["fraud_probability"]

    # Step 2: SHAP explanation
    shap_result = generate_explaination(scaled_df)

    shap_text = shap_result["explanation_text"]

    # Step 3: LLM explanation
    llm_explanation = genrate_llm_explanation(
        decision=decision,
        fraud_probability=fraud_probability,
        shap_explanations=shap_text
    )

    return{
        "decision": decision,
        "fraud_probability": fraud_probability,
        "shap_explanation": shap_text,
        "llm_explanation": llm_explanation
    }
