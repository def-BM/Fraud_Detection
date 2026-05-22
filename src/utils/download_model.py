from pathlib import Path
import os
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

def download_model():

    model_path = MODEL_DIR / "xgb_fraud_model.pkl"

    if not os.path.exists(model_path):

        os.makedirs(MODEL_DIR, exist_ok=True)

        url = "https://huggingface.co/bmaurya05/xplainable_fraud_detection/resolve/main/xgb_fraud_model.pkl"

        response = requests.get(url)

        with open(model_path, "wb") as f:
            f.write(response.content)

        print("Model downloaded")

    return model_path