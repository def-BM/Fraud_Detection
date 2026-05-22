# 🚀 Explainable Fraud Detection System

An advanced fraud detection system that not only predicts fraudulent transactions but also provides **human-understandable explanations** using Explainable AI and Generative AI.

This project integrates **Machine Learning + Explainability + RAG + LLMs** to make fraud detection **transparent, interpretable, and reliable**.

---

## Live Demo: https://xplainablefrauddetection.streamlit.app/

---

## 📌 Features

- 🔍 Fraud Detection using XGBoost
- 📊 Interactive dashboard with Streamlit
- 🧠 Explainability using SHAP & LIME
- 🤖 AI-generated explanations using LLM + RAG
- 📈 Real-time prediction system
- 📂 Structured pipeline
- 📉 Model performance evaluation

---

## 🛠️ Tech Stack

- Frontend/UI: Streamlit  
- Backend: Python  
- ML Model: XGBoost  
- Explainability: SHAP, LIME  
- Data Processing: Pandas, NumPy  
- Visualization: Matplotlib, Seaborn  
- Model Storage: Joblib  
- AI Layer: LLM + RAG  

---

## 📂 Project Structure
fraud-detection/
│
├── data/
| ├── processed_data.csv
│ └── raw_data.csv
│
├── models/
| ├── feature_columns.pkl
| ├── fraud_prototypes.pkl
│ ├── xgb_fraud.pkl
│ └── scaler.pkl
│
├── notebooks/
| ├── 1_data_exploration.ipynb
| ├── 2_explainability_shap.ipynb
│ └── 3_model_comp.ipynb
│
├── src/
│ ├── data_preprocessing/ 
| | └── preprocessing.py
│ ├── decision_engine/
| | └── fraud_decision.py
│ ├── explainers/
| | └── shap_explainer.py
| ├── llm/
| | └── llm_explainer.py
| ├── models/
| | ├── feature_mapper.py
| | ├── model_utils.py
| | └── train.py
│ └── app.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app/main.py

---

📊 Model Evaluation Metrics
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.88      0.83      0.85        98

    accuracy                           1.00     56962
   macro avg       0.94      0.91      0.93     56962
weighted avg       1.00      1.00      1.00     56962

Confusion Matrix:
 [[56853    11]
 [   17    81]]
ROC AUC Score: 0.9731
Precision-Recall AUC Score: 0.8753

⚠️ Fraud datasets are imbalanced, so Precision & Recall are more important than Accuracy.

---

🧠 Explainability
SHAP: Feature contribution analysis
LIME: Local explanation
LLM + RAG: Converts technical outputs into human-friendly explanations

---

🔄 Workflow
Input transaction data
Data preprocessing
Model prediction
Explainability (SHAP/LIME)
AI-generated explanation

---

📜 License

MIT License

---

## Author
Developer: Brijesh Maurya  
Final Year IT Engineering Student  
Domain Interests: Data Science, Machine Learning, Business Intelligence

