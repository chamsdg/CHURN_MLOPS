import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_data, clean_data, encode_data, FEATURES

# =====================
# MLFLOW CONFIG
# =====================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_pro")

client = MlflowClient()

MODEL_NAME = "ChurnModel"

# =====================
# DATA
# =====================
data = load_data("data/churn.csv")

X = data[FEATURES]
y = data["Exited"]

X = clean_data(X)
X, encoders = encode_data(X, fit=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# =====================
# MODELS
# =====================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight
    )
}

best_model = None
best_name = None
best_roc = -1
best_run_id = None

# =====================
# TRAIN + TRACK
# =====================
for name, model in models.items():

    with mlflow.start_run(run_name=name) as run:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_param("model", name)

        mlflow.sklearn.log_model(model, "model")

        if roc > best_roc:
            best_roc = roc
            best_model = model
            best_name = name
            best_run_id = run.info.run_id

# =====================
# BEST MODEL INFO
# =====================
print(f"\n🏆 Best model: {best_name} | ROC-AUC: {best_roc:.4f}")

# =====================
# REGISTER MODEL
# =====================
model_uri = f"runs:/{best_run_id}/model"

result = mlflow.register_model(
    model_uri,
    MODEL_NAME
)

version = result.version

print(f"📦 Registered model version: {version}")

# =====================
# STAGING AUTOMATION
# =====================
if best_roc > 0.80:   # threshold business rule

    print("🚀 Promoting to STAGING...")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging"
    )

    print("✅ Model moved to Staging")

    # OPTIONAL AUTO PROMOTION (business rule)
    if best_roc > 0.85:

        print("🔥 Promoting to PRODUCTION...")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production"
        )

        print("🚀 Model is now in Production")

else:
    print("⚠️ Model not good enough for staging")

# =====================
# LOCAL BACKUP (DOCKER SAFE)
# =====================
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "encoders": encoders,
    "model_name": best_name
}, "models/model_bundle.pkl")

print("✅ MLOps PRO pipeline completed")