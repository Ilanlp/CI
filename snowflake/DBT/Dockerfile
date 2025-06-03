FROM python:3.10.17-slim

# Créer un utilisateur non-root
RUN groupadd -r node && useradd -r -g node node

# Variables d’environnement utiles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DBT_PARTIAL_PARSE=0

# Dossier de travail principal
WORKDIR /usr/src/DBT

# Copier les dépendances
COPY requirements.txt ./

# Installer les dépendances système + Python
RUN apt-get update -y && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier le projet (en root pour gérer les permissions juste après)
COPY Projet_DBT/ Projet_DBT/

# Supprimer target s’il existe, le recréer avec droits complets
RUN rm -rf /usr/src/DBT/Projet_DBT/target && \
    mkdir -p /usr/src/DBT/Projet_DBT/target && \
    chmod -R 777 /usr/src/DBT/Projet_DBT && \
    chmod -R 777 /usr/src/DBT/Projet_DBT/target

# Créer le dossier .dbt avec droits complets aussi
RUN mkdir -p /home/node/.dbt && \
    chmod -R 777 /home/node/.dbt

# Changer d’utilisateur
USER node
WORKDIR /usr/src/DBT/Projet_DBT

# Lancer config + dbt
CMD mkdir -p target && chmod -R 777 target && \
    python3 config.py && \
    dbt debug --profiles-dir /home/node/.dbt && \
    dbt run --profiles-dir /home/node/.dbt --select bronze && \
    dbt run --profiles-dir /home/node/.dbt --select silver && \
    dbt run --profiles-dir /home/node/.dbt --select liaison_Rome_Metier_gold_sql liaison_Rome_Soft_Skill_gold_sql && \
    dbt run --profiles-dir /home/node/.dbt --select fait_offre && \
    dbt run --profiles-dir /home/node/.dbt --select liaison_Offre_Competence
