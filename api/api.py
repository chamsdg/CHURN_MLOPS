from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import mlflow.pyfunc

# =====================
# APP
# =====================
app = FastAPI()

# =====================
# LOAD DATASET
# =====================
data = pd.read_csv("data/churn.csv")

# =====================
# LOAD FEATURES + ENCODERS (LOCAL SAFE)
# =====================
import joblib
bundle = joblib.load("models/model_bundle.pkl")

features = bundle["features"]
label_encoders = bundle["encoders"]

# =====================
# LOAD MODEL FROM MLFLOW REGISTRY (PRO MODE)
# =====================
MODEL_NAME = "ChurnModel"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

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
    return {
        "status": "API running (MLflow Production mode)",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE
    }

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

        # prediction MLflow
        proba = model.predict(df)[0]

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
            "risk": risk,
            "model_source": "mlflow_registry"
        }

    except Exception as e:
        logging.error(e)
        return {"error": str(e)}