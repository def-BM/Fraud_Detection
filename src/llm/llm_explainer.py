import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# try local .env first
api_key = os.getenv("GROQ_API_KEY")

# If not found, try Streamlit secrets
if not api_key:
    api_key = st.secrets.get("GROQ_API_KEY")

client = Groq(api_key=api_key)

def genrate_llm_explanation(
        decision: str,
        fraud_probability: float,
        shap_explanations: list
):
    """
    Converts SHAP explanations into human-like fraud analyst explaination
    """

    prompt = f"""
You are senior financial fraud analyst.

Transaction Decision: {decision}
Fraud Probability: {fraud_probability}

Key Risk Factors:
{chr(10).join(shap_explanations)}

Explain clearly:
1. Why the transaction was flagged or allowed
2. Which factors influenced the decision
3. Whether the transaction seems genuinely fraudulent

Keep explanation concise and professional.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content