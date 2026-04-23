from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import joblib

# =====================
# APP
# =====================
app = FastAPI()

# =====================
# LOAD DATASET
# =====================
data = pd.read_csv("data/churn.csv")

# =====================
# LOAD BUNDLE (SAFE DOCKER WAY)
# =====================
bundle = joblib.load("models/model_bundle.pkl")

model = bundle["model"]
features = bundle["features"]
label_encoders = bundle["encoders"]

# =====================
# LOGGING
# =====================
logging.basicConfig(level=logging.INFO)

# =====================
# INPUT
# =====================
class PredictionInput(BaseModel):
    CustomerId: int

# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"status": "API running (Docker safe mode)"}

# =====================
# PREDICT
# =====================
@app.post("/predict")
def predict(input_data: PredictionInput):

    try:
        # récupérer client réel
        client = data[data["CustomerId"] == input_data.CustomerId]

        if client.empty:
            return {"error": "CustomerId introuvable"}

        df = client[features].copy()

        # encoding
        for col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

        # prediction
        proba = model.predict_proba(df)[0][1]

        prediction = int(proba > 0.5)

        # risk score
        if proba < 0.4:
            risk = "Low"
        elif proba < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "CustomerId": input_data.CustomerId,
            "prediction": prediction,
            "probability": float(proba),
            "risk": risk
        }

    except Exception as e:
        logging.error(e)
        return {"error": str(e)}