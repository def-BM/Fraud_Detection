from pathlib import Path
import os
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

FILES = {
    "xgb_fraud_model.pkl":
    "https://huggingface.co/bmaurya05/xplainable_fraud_detection/resolve/main/xgb_fraud_model.pkl",

    "feature_columns.pkl":
    "https://huggingface.co/bmaurya05/xplainable_fraud_detection/resolve/main/feature_columns.pkl",

    "fraud_prototypes.pkl":
    "https://huggingface.co/bmaurya05/xplainable_fraud_detection/resolve/main/fraud_prototypes.pkl",

    "scaler.pkl":
    "https://huggingface.co/bmaurya05/xplainable_fraud_detection/resolve/main/scaler.pkl"
}

def download_files():

    MODEL_DIR.mkdir(exist_ok=True)

    for filename, url in FILES.items():

        file_path = MODEL_DIR / filename

        if not file_path.exists():

            response = requests.get(url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"{filename} downloaded")