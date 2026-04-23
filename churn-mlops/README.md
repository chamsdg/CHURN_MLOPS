# 🚀 CHURN MLOPS PROJECT - GUIDE COMPLET

## 📌 Objectif

Construire un pipeline MLOps complet pour prédire le churn client avec :

* Machine Learning
* API FastAPI
* MLflow (tracking + registry)
* Docker
* CI/CD (GitHub Actions)
* Déploiement production

---

# 📁 Structure du projet

```
CHURN_MLOPS/
│
├── api/                # API FastAPI
│   └── api.py
│
├── src/                # Code ML
│   ├── train.py
│   └── preprocess.py
│
├── data/               # Dataset
│   └── churn.csv
│
├── models/             # Modèle sauvegardé
│   └── model_bundle.pkl
│
├── mlruns/             # MLflow tracking
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── .github/workflows/mlops.yml
```

---

# ⚙️ 1. Entraînement du modèle

## ▶️ Lancer le training

```
python src/train.py
```

## 🔥 Ce que fait le script :

* Chargement des données
* Nettoyage (NaN)
* Encodage (LabelEncoder)
* Split train/test
* Gestion du déséquilibre (`scale_pos_weight`)
* Entraînement de plusieurs modèles :

  * Logistic Regression
  * Random Forest
  * XGBoost
  * LightGBM
* Évaluation (Accuracy + ROC-AUC)
* Sélection du meilleur modèle
* Enregistrement MLflow
* Sauvegarde locale (`model_bundle.pkl`)

---

# 📊 2. MLflow

## ▶️ Lancer MLflow UI

```
mlflow ui
```

Puis ouvrir :

```
http://localhost:5000
```

## 🔥 Fonctionnalités :

* Comparer les modèles
* Voir les métriques
* Registry (Staging / Production)

---

# 🚀 3. API FastAPI

## ▶️ Lancer localement

```
uvicorn api.api:app --reload
```

## 📌 Endpoint principal

```
POST /predict
```

### Exemple :

```
{
  "CustomerId": 15737452
}
```

### Réponse :

```
{
  "CustomerId": 15737452,
  "prediction": 1,
  "probability": 0.73,
  "risk": "High"
}
```

---

# 🐳 4. Docker

## ▶️ Build image

```
docker build -t churn-api .
```

## ▶️ Run container

```
docker run -p 8000:8000 churn-api
```

## 🔎 Test

```
http://localhost:8000/docs
```

---

# ☁️ 5. Docker Hub

## ▶️ Tag image

```
docker tag churn-api aidarachams/churn-api:latest
```

## ▶️ Push

```
docker push aidarachams/churn-api:latest
```

---

# 🔁 6. CI - GitHub Actions

## 📌 Fichier :

```
.github/workflows/mlops.yml
```

## 🔥 Pipeline :

* Installer dépendances
* Lancer entraînement
* Vérifier modèle
* Build Docker

---

# 🌍 7. Déploiement (Render)

## Étapes :

1. Créer compte Render
2. Connecter GitHub
3. Créer Web Service
4. Choisir Docker
5. Déployer

## Résultat :

```
https://your-api.onrender.com
```

---

# 🔄 8. Pipeline MLOps complet

```
Code → GitHub → CI → Train → MLflow → Docker → Deploy → API
```

---

# 🧠 Bonnes pratiques

## ✅ Toujours :

* Versionner le code (Git)
* Tracker les modèles (MLflow)
* Tester en local (Docker)
* Automatiser (CI/CD)

## ❌ Éviter :

* pousser venv/
* pousser mlruns/
* images Docker trop lourdes

---

# 🔮 Améliorations futures

* Monitoring (logs, drift)
* Auto retraining
* Kubernetes deployment
* A/B testing modèle
* Feature store

---

# 👨‍💻 Auteur

Projet MLOps complet – niveau professionnel 🚀
