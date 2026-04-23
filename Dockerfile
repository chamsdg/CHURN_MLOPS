FROM python:3.10-slim

# =====================
# ENV
# =====================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =====================
# WORKDIR
# =====================
WORKDIR /app

# =====================
# SYSTEM DEPENDENCIES
# =====================
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =====================
# INSTALL PYTHON DEPENDENCIES
# =====================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================
# COPY PROJECT
# =====================
COPY . .

# =====================
# CREATE MODELS FOLDER (IMPORTANT)
# =====================
RUN mkdir -p models

# =====================
# EXPOSE API PORT
# =====================
EXPOSE 8000

# =====================
# START API
# =====================
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]