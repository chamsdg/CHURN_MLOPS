import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_data, clean_data, encode_data, FEATURES

# =====================
# REPRODUCIBILITY
# =====================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# =====================
# MLFLOW SETUP
# =====================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_prediction")

# =====================
# LOAD DATA
# =====================
data = load_data("data/churn.csv")

X = data[FEATURES]
y = data["Exited"]

# =====================
# PREPROCESSING
# =====================
X = clean_data(X)
X, encoders = encode_data(X, fit=True)

# =====================
# TRAIN / TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# =====================
# CLASS BALANCE
# =====================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# =====================
# MODELS
# =====================
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED
    )
}

# =====================
# TRAINING + TRACKING
# =====================
best_model = None
best_name = None
best_score = -1

for name, model in models.items():

    print(f"\n===== Training {name} =====")

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred))

        # =====================
        # LOG MLFLOW
        # =====================
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_param("model_name", name)
        mlflow.log_param("seed", SEED)

        mlflow.sklearn.log_model(model, "model")

        # =====================
        # BEST MODEL SELECTION
        # =====================
        if roc > best_score:
            best_score = roc
            best_model = model
            best_name = name

# =====================
# BEST MODEL INFO
# =====================
print(f"\n🏆 Best model: {best_name}")
print(f"Best ROC-AUC: {best_score:.4f}")

# =====================
# REGISTER IN MLFLOW REGISTRY
# =====================
with mlflow.start_run(run_name="REGISTER_BEST_MODEL"):

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_roc_auc", best_score)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="ChurnModel"
    )

# =====================
# SAVE LOCAL BACKUP (API)
# =====================
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "features": FEATURES,
    "encoders": encoders
}, "models/model_bundle.pkl")

print("✅ MLOps pipeline completed successfully")
print("🚀 Model registered in MLflow + saved locally")