# 🚀 Guide MLOps – Projet Churn (MLflow + GitHub Actions + Docker + FastAPI)

## 📌 1. Objectif du projet

Ce projet a pour objectif de construire un système complet de Machine Learning en production (MLOps) capable de :

- prédire le churn client (désabonnement)
- entraîner plusieurs modèles ML
- suivre les expériences
- déployer un modèle en API
- automatiser le pipeline avec CI/CD

---

# 🧠 2. Qu’est-ce que le MLOps ?

Le **MLOps** (Machine Learning Operations) est une discipline qui permet de :

- industrialiser les modèles ML
- automatiser leur entraînement
- suivre leurs performances
- les déployer en production

👉 Objectif : passer du notebook → production

---

# 📊 3. Pipeline du projet

```text
Data → Preprocessing → Training → MLflow → Registry → API → Docker → CI/CD
```

---

# 🧪 4. MLflow (Tracking + Registry)

entity["software","MLflow","machine learning lifecycle tool"]

## 📌 Rôle de MLflow

MLflow permet de gérer tout le cycle de vie du modèle ML :

### 1. Tracking
- enregistre les expériences
- sauvegarde les métriques (accuracy, ROC-AUC)
- compare les modèles

### 2. Registry
- stocke les modèles versionnés
- permet de gérer :
  - Staging
  - Production

---

## 📌 Exemple de workflow MLflow

```text
Train model → log metrics → save model → register model
```

---

## 📌 Staging vs Production

- **Staging** : modèle testé
- **Production** : modèle utilisé par l’API

👉 On peut promouvoir manuellement ou automatiquement un modèle.

---

# ⚙️ 5. GitHub Actions (CI/CD)

## 📌 Définition
GitHub Actions permet d’automatiser les tâches du projet.

## 📌 Dans ce projet

À chaque push sur `master` :

### 1. Training automatique
- installation dépendances
- entraînement des modèles
- génération MLflow tracking

### 2. Artifact upload
- sauvegarde des résultats MLflow
- sauvegarde du modèle

### 3. Docker build
- création image API

---

## 📌 Pipeline CI/CD

```text
GitHub Push → Train Model → Save Artifact → Build Docker
```

---

# 🐳 6. Docker

Docker permet de containeriser l’application.

## 📌 Rôle

- isoler l’environnement
- garantir reproductibilité
- déployer partout

## 📌 Étapes

- installation dépendances
- copie du code
- lancement FastAPI

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000
```

---

# ⚡ 7. FastAPI (API de prédiction)

## 📌 Rôle

Expose le modèle ML via une API REST.

## 📌 Fonctionnement

1. reçoit CustomerId
2. récupère données client
3. encode les features
4. appelle modèle
5. retourne prediction + probabilité

---

# 🔁 8. Model Registry (important)

Le **Model Registry MLflow** permet de :

- versionner les modèles
- stocker plusieurs versions
- gérer staging/production

## 📌 Exemple

```text
ChurnModel
 ├── Version 1 (Staging)
 ├── Version 2 (Production)
```

---

# 🚀 9. Architecture finale

```text
GitHub Actions
      ↓
Training (MLflow)
      ↓
Model Registry
      ↓
Staging → Production
      ↓
FastAPI
      ↓
Docker
```

---

# 💡 10. Concepts clés appris

✔ Machine Learning pipeline
✔ Model versioning
✔ CI/CD
✔ Dockerization
✔ API deployment
✔ MLflow tracking
✔ MLflow registry

---

# 🏆 11. Niveau atteint

Tu es passé de :

```text
Notebook ML simple
```

à :

```text
MLOps pipeline complet (production-ready)
```

---

# 🚀 12. Prochaines améliorations possibles

- MLflow server (Docker)
- Monitoring drift
- A/B testing models
- Auto deployment cloud (AWS / Render)
- Feature store

---

# 🎯 Conclusion

Ce projet représente une **architecture MLOps complète** incluant :

- entraînement
- tracking
- registry
- API
- CI/CD
- containerisation

👉 C’est une base solide pour un projet industriel ML.

