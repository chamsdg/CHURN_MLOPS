import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Charger données
data = pd.read_csv('churn.csv')

# Features
features = [
    'RowNumber', 'CustomerId', 'CreditScore',
    'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]

target = 'Exited'

X = data[features].copy()
y = data[target]

# NaN
X = X.ffill()

# Encodage
label_encoders = {}
for col in ['Geography', 'Gender']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Déséquilibre
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# =====================
# MODELS
# =====================

models = {

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
}

# =====================
# TRAIN & EVALUATE
# =====================

results = {}

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))

    results[name] = roc  # 🔥 on choisit avec ROC-AUC

# =====================
# BEST MODEL
# =====================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Best model: {best_model_name}")

# =====================
# SAVE (version enrichie)
# =====================

joblib.dump({
    "model": best_model,
    "model_name": best_model_name,
    "features": features,
    "encoders": label_encoders
}, 'model_bundle.pkl')

print("✅ Modèle complet sauvegardé (bundle)")