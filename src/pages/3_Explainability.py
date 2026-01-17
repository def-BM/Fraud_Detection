import streamlit as st

st.title("🧠 Explainability")

st.markdown("""
### 🔍 SHAP Explanation
These features influenced the fraud decision:
""")

# This will be stored in session state after prediction
if "last_result" in st.session_state:
    for exp in st.session_state["last_result"]["shap_explanation"]:
        st.write("•", exp)
    
    st.markdown("---")
    st.markdown("### 🤖 GenAI Explanation")
    st.write(st.session_state["last_result"]["llm_explanation"])
else:
    st.warning("Run a fraud check first.")