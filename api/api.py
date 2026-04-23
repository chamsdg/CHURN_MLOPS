from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import mlflow.pyfunc
import joblib
import os

# =====================
# APP
# =====================
app = FastAPI()

# =====================
# LOGGING
# =====================
logging.basicConfig(level=logging.INFO)

# =====================
# LOAD DATASET
# =====================
data = pd.read_csv("data/churn.csv")

# =====================
# LOAD BUNDLE (LOCAL FALLBACK)
# =====================
bundle = joblib.load("models/model_bundle.pkl")

features = bundle["features"]
label_encoders = bundle["encoders"]

# =====================
# MODEL CONFIG
# =====================
MODEL_NAME = "ChurnModel"
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# =====================
# LOAD MODEL (MLFLOW + FALLBACK)
# =====================
try:
    model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    MODEL_SOURCE = "mlflow_registry"

except Exception as e:
    logging.warning(f"MLflow model load failed, fallback local model: {e}")

    model = bundle["model"]
    MODEL_SOURCE = "local_bundle"

# =====================
# INPUT
# =====================
class PredictionInput(BaseModel):
    CustomerId: int

# =====================
# HEALTH CHECK
# =====================
@app.get("/health")
def health():
    return {
        "status": "API running",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "source": MODEL_SOURCE
    }

# =====================
# PREDICT
# =====================
@app.post("/predict")
def predict(input_data: PredictionInput):

    try:
        # get client
        client = data[data["CustomerId"] == input_data.CustomerId]

        if client.empty:
            return {"error": "CustomerId introuvable"}

        df = client[features].copy()

        # encoding
        for col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

        # prediction
        proba = model.predict(df)[0]

        prediction = int(proba > 0.5)

        # risk scoring
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
            "risk": risk,
            "model_source": MODEL_SOURCE
        }

    except Exception as e:
        logging.error(e)
        return {"error": str(e)}