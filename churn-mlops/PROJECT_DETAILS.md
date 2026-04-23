# Projet Churn MLOps - Explication détaillée

## Vue d'ensemble
Le projet Churn MLOps est conçu pour prédire l'attrition des clients en utilisant des techniques d'apprentissage automatique et en mettant en œuvre les principes de MLOps pour un développement, un déploiement et une surveillance efficaces des modèles. Le projet est structuré pour garantir évolutivité, maintenabilité et facilité d'utilisation.

---

## Structure du projet

```
churn-mlops/
│
├── data/
│   └── churn.csv          # Jeu de données contenant les informations des clients et les étiquettes d'attrition
│
├── models/
│   └── model_bundle.pkl   # Modèle d'apprentissage automatique entraîné
│
├── src/
│   ├── train.py           # Script pour entraîner le modèle
│   ├── preprocess.py      # Script pour prétraiter les données
│
├── api/
│   └── main.py            # Application FastAPI pour fournir des prédictions
│
├── logs/
│   └── predictions.log    # Fichier de journalisation des prédictions de l'API
│
├── requirements.txt       # Liste des dépendances du projet
└── README.md              # Documentation générale du projet
```

---

## Explication détaillée de chaque composant

### 1. **Données**
- **Fichier** : `data/churn.csv`
- Ce fichier contient le jeu de données utilisé pour entraîner le modèle de prédiction d'attrition. Il inclut des informations sur les clients telles que :
  - `RowNumber` : Index du client.
  - `CustomerId` : Identifiant unique pour chaque client.
  - `Surname` : Nom de famille du client.
  - `CreditScore` : Score de crédit du client.
  - `Geography` : Pays du client.
  - `Gender` : Genre du client.
  - `Age` : Âge du client.
  - `Tenure` : Nombre d'années pendant lesquelles le client est resté avec l'entreprise.
  - `Balance` : Solde du compte du client.
  - `NumOfProducts` : Nombre de produits détenus par le client.
  - `HasCrCard` : Si le client possède une carte de crédit (1 = Oui, 0 = Non).
  - `IsActiveMember` : Si le client est un membre actif (1 = Oui, 0 = Non).
  - `EstimatedSalary` : Salaire estimé du client.
  - `Exited` : Si le client a quitté l'entreprise (1 = Oui, 0 = Non).

### 2. **Modèles**
- **Fichier** : `models/model_bundle.pkl`
- Ce fichier contient le modèle d'apprentissage automatique entraîné (par exemple, un `RandomForestClassifier`) sérialisé à l'aide de `joblib`. Il est utilisé par l'API pour effectuer des prédictions.

### 3. **Code source**
#### a. `src/train.py`
- Ce script est responsable de l'entraînement du modèle d'apprentissage automatique.
- Il effectue les étapes suivantes :
  1. Charge le jeu de données depuis `data/churn.csv`.
  2. Prétraite les données (par exemple, encodage des variables catégoriques).
  3. Divise les données en ensembles d'entraînement et de test.
  4. Entraîne un modèle `RandomForestClassifier`.
  5. Sauvegarde le modèle entraîné dans `models/model_bundle.pkl`.

#### b. `src/preprocess.py`
- Ce script gère les tâches de prétraitement des données, telles que :
  1. Chargement du jeu de données.
  2. Encodage des variables catégoriques (par exemple, `Geography`, `Gender`) à l'aide de `LabelEncoder`.
  3. Retourne les données prétraitées et les encodeurs pour l'entraînement et la prédiction.

### 4. **API**
- **Fichier** : `api/main.py`
- Cette application FastAPI sert le modèle entraîné pour effectuer des prédictions.
- **Endpoints** :
  - `GET /health` : Vérifie si l'API fonctionne.
  - `POST /predict` : Accepte un `CustomerId` en entrée et retourne une prédiction d'attrition.
- **Journalisation** :
  - Toutes les prédictions sont enregistrées dans `logs/predictions.log` pour le suivi et le débogage.

### 5. **Logs**
- **Fichier** : `logs/predictions.log`
- Ce fichier stocke les journaux de toutes les prédictions effectuées par l'API, y compris les données d'entrée et les résultats des prédictions.

### 6. **Dépendances**
- **Fichier** : `requirements.txt`
- Ce fichier répertorie toutes les dépendances Python nécessaires au projet, notamment :
  - `fastapi` : Pour la création de l'API.
  - `uvicorn` : Pour exécuter le serveur FastAPI.
  - `pandas` : Pour la manipulation des données.
  - `scikit-learn` : Pour l'apprentissage automatique.
  - `joblib` : Pour sauvegarder et charger le modèle entraîné.

### 7. **Documentation**
- **Fichier** : `README.md`
- Fournit une vue d'ensemble du projet, sa structure et des instructions pour l'exécuter.

---

## Étapes pour exécuter le projet

1. **Cloner le dépôt** :
   ```bash
   git clone <repository-url>
   cd churn-mlops
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Entraîner le modèle** :
   ```bash
   python src/train.py
   ```

4. **Démarrer l'API** :
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Tester l'API** :
   - Utilisez des outils comme Postman ou cURL pour envoyer des requêtes aux endpoints.
   - Exemple de requête pour l'endpoint `/predict` :
     ```json
     {
       "CustomerId": 15634602
     }
     ```

6. **Surveiller les logs** :
   - Consultez `logs/predictions.log` pour un enregistrement de toutes les prédictions effectuées par l'API.

---

## Améliorations futures
- **Pipeline CI/CD** : Automatiser les tests, l'entraînement et le déploiement à l'aide de GitHub Actions.
- **Surveillance du modèle** : Intégrer des outils comme MLflow ou Prometheus pour surveiller les performances du modèle.
- **Conteneurisation** : Utiliser Docker pour conteneuriser l'application et faciliter son déploiement.
- **Déploiement dans le cloud** : Déployer l'application sur une plateforme cloud comme AWS, Azure ou GCP.

---

Ce projet est conçu pour être extensible et peut être adapté à d'autres cas d'utilisation nécessitant des prédictions basées sur des modèles d'apprentissage automatique.