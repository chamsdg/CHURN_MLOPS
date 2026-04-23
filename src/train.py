import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_data, clean_data, encode_data, FEATURES

# =====================
# MLFLOW CONFIG PRO
# =====================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("churn_pro")

# =====================
# DATA
# =====================
data = load_data("data/churn.csv")

X = data[FEATURES]
y = data["Exited"]

X = clean_data(X)
X, encoders = encode_data(X, fit=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

# =====================
# TRAIN + LOG
# =====================
for name, model in models.items():

    with mlflow.start_run(run_name=name):

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

# =====================
# REGISTER MODEL (PRO STEP)
# =====================
with mlflow.start_run(run_name="register_best"):

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_roc", best_roc)

    mlflow.sklearn.log_model(
        best_model,
        "model",
        registered_model_name="ChurnModel"
    )

# =====================
# SAVE LOCAL BUNDLE
# =====================
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "encoders": encoders,
    "model_name": best_name
}, "models/model_bundle.pkl")

print("✅ PRO MODEL TRAINED + REGISTERED")