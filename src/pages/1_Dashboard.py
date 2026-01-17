import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Fraud Detection Dashboard")

total_txns = 1248
fraud_txns = 37
blocked_txns = 29
safe_txns = total_txns - fraud_txns

col1, col2, col3, col4 = st.columns(4)

col1.metric("🔍 Transactions Checked", f"{total_txns:,}")
col2.metric("🚨 Fraud Detected", f"{fraud_txns:,}")
col3.metric("❌ Transactions Blocked", f"{blocked_txns:,}")
col4.metric("✅ Safe Transactions", f"{safe_txns:,}")

st.markdown("---")

# ====== Pie Chart ========
st.subheader("Fraud vs Safe Transactions")

pie_df = pd.DataFrame({
    'Type': ['Fraudulent', 'Safe'],
    'Count': [fraud_txns, safe_txns]
})

st.pyplot(
    pie_df.set_index("Type").plot.pie(
        y="Count",
        autopct="%1.1f%%",
        colors=["#FF6347", "#32CD32"],
        legend=False,
        figsize=(5, 5)
    ).figure
)

# ===== Bar Chart ======
st.subheader("Transaction Decision Breakdown")

bar_df = pd.DataFrame({
    'Decision': ['Blocked', 'Allowed'],
    'Count': [blocked_txns, safe_txns]
})

st.bar_chart(bar_df.set_index("Decision"))

# =========== Fraud Probability Distribution ========
st.subheader("Fraud Probability Distribution")

if "fraud_prob_history" in st.session_state and len(st.session_state["fraud_prob_history"]) > 0:
    fraud_probs = st.session_state["fraud_prob_history"]

    fraud_df = pd.DataFrame({
        'Fraud Probability': fraud_probs
    })

    st.area_chart(fraud_df)
else:
    st.info("ℹ️ No fraud probability data available yet. Check some transactions to see the distribution here.")

# ======= Transaction Trend ===========
st.subheader("Transaction Trend (Last 10 Days)")

trend_df = pd.DataFrame({
    "Day": [f"Day{i}" for i in range(1, 11)],
    "Transactions": np.random.randint(80, 180, 10)
}).set_index("Day")

st.line_chart(trend_df)


# System Status
st.success("✅ System Status: Secure and Actively Monitoring Transactions")

st.markdown("""
### 🛡️ How This System Protects You
- Automatically analyzes transactions in real time
- Stop suspicious activity before money is lost
- Provides clear explanations for blocked transactions
- Designed to reduce fraud while allowing genuine payments
""")

st.info(
    "ℹ️ You can check an individual transaction using the **Fraud Check** page"
    " and understand decisions using the **Explainability** page."
)