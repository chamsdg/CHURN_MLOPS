FROM python:3.10-slim

# dossier de travail
WORKDIR /app

# dépendances système (important pour lightgbm/xgboost)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copier requirements
COPY requirements.txt .

# installer dépendances
RUN pip install --no-cache-dir -r requirements.txt

# copier projet
COPY . .

# exposer API
EXPOSE 8000

# lancer API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]