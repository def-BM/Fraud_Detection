import streamlit as st
from pipeline.run_fraud_pipeline import run_fraud_pipeline
from models.model_utils import build_behavioral_features
from models.feature_mapper import map_to_model_features

st.title("🚨 Fraud Transaction Check")
st.markdown("Enter transaction details:")

# =========== User Inputs ===========
amount = st.number_input("Transaction Amount (₹)", min_value=0.0)
txn_hour = st.slider("Transaction Hour", 0, 23, 12)
location_risk = st.selectbox("Location Risk Level", ["Low", "Medium", "High"])
device_type = st.selectbox("Device Type", ["Known Device", "New Device"])
velocity = st.slider("Transaction Velocity (transactions in last 1 hour)", 0, 20, 1)

# Initialize session state for fraud probability history
if "fraud_prob_history" not in st.session_state:
    st.session_state["fraud_prob_history"] = []

# ============= Prediction =============
if st.button("🔍 Check for Fraud"):
    behavioral = build_behavioral_features(
        amount, txn_hour, velocity, location_risk, device_type
    )

    # Model expects 30 features
    model_input = map_to_model_features(behavioral)

    result = run_fraud_pipeline(model_input)
    st.session_state['last_result'] = result
    
    # store real fraud probability
    st.session_state["fraud_prob_history"].append(
        result["fraud_probability"]
    )

    st.subheader("🧠 Fraud Analysis Result")

    decision = result["decision"].upper()

    if decision == "BLOCK":
        st.error("🚨 Fraud Detected — Transaction Blocked")
    else:
        st.success("✅ Transaction Safe")
    
    st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")

