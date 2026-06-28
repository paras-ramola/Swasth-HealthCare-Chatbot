import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "disease_prediction_model.cbm":
    "https://huggingface.co/ParSuper/Disease_Prediction/resolve/main/disease_prediction_model.cbm",

    "symptom_columns.pkl":
    "https://huggingface.co/ParSuper/Disease_Prediction/resolve/main/symptom_columns.pkl",

    "disease_classes.pkl":
    "https://huggingface.co/ParSuper/Disease_Prediction/resolve/main/disease_classes.pkl"
}

for filename, url in FILES.items():
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        r = requests.get(url)

        with open(filepath, "wb") as f:
            f.write(r.content)

print("Models ready.")