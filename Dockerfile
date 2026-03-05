# Utiliser une version stable de Python (3.11 est recommandée pour l'IA)
FROM python:3.12.3

# Définir le répertoire de travail dans le container
WORKDIR /

# Installer les dépendances système nécessaires (ex: pour ydata-profiling ou psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY . .

# Exposer le port (8000 pour FastAPI, 8501 pour Streamlit)
EXPOSE 8000 8501

# Commande de lancement de l'api et de l'app streamlit
CMD uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.address 0.0.0.0 --server.port 8501    