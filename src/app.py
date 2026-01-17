import streamlit as st

st.set_page_config(
    page_title="Explainable Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Explainable Fraud Detection System")
st.markdown("""
            Welcome to the **Explainable Fraud Detection Platform**
            Use the sidebar to nevigate between pages.
            """)
st.sidebar.success("Select a page above 👆")
