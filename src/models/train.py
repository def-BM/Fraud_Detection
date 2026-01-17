import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Paths
DATA_DIR = 'data/processed_data/'
MODEL_DIR = 'models/'

os.makedirs(MODEL_DIR, exist_ok=True)

# Load Processed Data
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    print("Processed data loaded successfully.")
    return X_train, X_test, y_train, y_test

# Train Model
def train_model(X_train, y_train):
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=fraud_ratio,
        eval_metric='auc',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nClassificatin Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC AUC Score: {auc:.4f}")

# Save Model
def save_model(model):
    model_path = os.path.join(MODEL_DIR, 'xgb_fraud_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

# Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    print("Training pipeline completed.")