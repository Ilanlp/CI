#!/bin/bash

echo " [1/3] Build des services ETL (sans les lancer)..."
docker-compose --profile etl build

echo " [2/3] Build des autres services (profil par défaut)..."
docker-compose --profile airflow build
docker-compose build

echo " [3/3] Lancement des services principaux..."
docker-compose --profile airflow up -d

echo " Tous les services standards sont démarrés."
echo " Les services ETL sont buildés et seront déclenchés par Airflow via DockerOperator."

