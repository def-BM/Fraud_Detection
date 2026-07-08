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
```text
fraud-detection/
│
├── data/
│   ├── processed_data.csv
│   └── raw_data.csv
│
├── models/
│   ├── feature_columns.pkl
│   ├── fraud_prototypes.pkl
│   ├── xgb_fraud.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_explainability_shap.ipynb
│   └── 3_model_comp.ipynb
│
├── src/
│   ├── data_preprocessing/
│   │   └── preprocessing.py
│   │
│   ├── decision_engine/
│   │   └── fraud_decision.py
│   │
│   ├── explainers/
│   │   └── shap_explainer.py
│   │
│   ├── llm/
│   │   └── llm_explainer.py
│   │
│   ├── models/
│   │   ├── feature_mapper.py
│   │   ├── model_utils.py
│   │   └── train.py
│   │
│   └── app.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

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

## 📊 Model Evaluation Metrics
## Classification Report

| Metric | Class 0 | Class 1 |
|---------|---------|---------|
| Precision | 1.00 | 0.88 |
| Recall | 1.00 | 0.83 |
| F1-Score | 1.00 | 0.85 |

### Overall Performance

| Metric | Score |
|---------|------:|
| Accuracy | 1.00 |
| Macro Precision | 0.94 |
| Macro Recall | 0.91 |
| Macro F1-Score | 0.93 |
| Weighted F1-Score | 1.00 |

---

## Confusion Matrix

```text
                Predicted
              Normal  Fraud
Actual Normal  56853     11
Actual Fraud      17     81
```

---

## Performance Metrics

| Metric | Score |
|---------|------:|
| ROC-AUC | **0.9731** |
| Precision-Recall AUC | **0.8753** |

> **Note:** Fraud detection datasets are highly imbalanced. Therefore, **Precision**, **Recall**, **F1-Score**, and **PR-AUC** provide a more reliable evaluation than Accuracy alone.

---

## 🧠 Explainability
SHAP: Feature contribution analysis
LIME: Local explanation
LLM + RAG: Converts technical outputs into human-friendly explanations

---

## 🔄 Workflow
Input transaction data
Data preprocessing
Model prediction
Explainability (SHAP/LIME)
AI-generated explanation

---

## 📜 License

MIT License

---

## Author
Developer: Brijesh Maurya  
Final Year IT Engineering Student  
Domain Interests: Data Science, Machine Learning, Business Intelligence

