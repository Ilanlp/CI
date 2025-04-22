"""
Collecteur de données d'emploi pour Adzuna
Ce script utilise l'API Adzuna pour récupérer les offres d'emploi
et les sauvegarder dans un fichier CSV.
"""

import asyncio
import pandas as pd
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from os import environ
from pathlib import Path
import time
import csv

# Import des clients API
from adzuna import AdzunaClient, CountryCode, SortBy
from jm_normalizer import JobDataNormalizer, NormalizedJobOffer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Liste des catégories Adzuna à collecter
ADZUNA_CATEGORIES = [
    "it-jobs",           # Informatique
    "engineering-jobs",  # Ingénierie
    "scientific-qa-jobs", # Scientifique et QA
    "creative-design-jobs", # Créatif et Design
    "consultancy-jobs",  # Conseil
    "graduate-jobs"      # Jeunes diplômés
]

async def main():
    """Fonction principale"""
    # Charger les variables d'environnement depuis un fichier .env si disponible
    load_dotenv()

    # Récupérer les identifiants d'API depuis les variables d'environnement
    adzuna_app_id = environ.get("ADZUNA_APP_ID")
    adzuna_app_key = environ.get("ADZUNA_APP_KEY")
    france_travail_id = environ.get("FRANCE_TRAVAIL_ID")
    france_travail_key = environ.get("FRANCE_TRAVAIL_KEY")

    # Vérifier que les variables d'environnement sont définies
    if not all([adzuna_app_id, adzuna_app_key, france_travail_id, france_travail_key]):
        missing_vars = []
        if not adzuna_app_id:
            missing_vars.append("ADZUNA_APP_ID")
        if not adzuna_app_key:
            missing_vars.append("ADZUNA_APP_KEY")
        if not france_travail_id:
            missing_vars.append("FRANCE_TRAVAIL_ID")
        if not france_travail_key:
            missing_vars.append("FRANCE_TRAVAIL_KEY")

        raise ValueError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}. "
            "Veuillez définir ces variables dans un fichier .env ou dans votre environnement."
        )

    # Créer le dossier de sortie si nécessaire
    output = environ.get("OUTPUT_DIR", "data/")
    path_absolu = Path(__file__).resolve()
    output_dir = f"{path_absolu.parents[2]}/{output}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialiser l'API Adzuna et le normalisateur
    adzuna_client = AdzunaClient(adzuna_app_id, adzuna_app_key)
    normalizer = JobDataNormalizer(
        adzuna_app_id,
        adzuna_app_key,
        france_travail_id,
        france_travail_key
    )

    # Créer le fichier CSV de sortie
    output_file = os.path.join(output_dir, f"adzuna_jobs_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Obtenir les noms des colonnes à partir du modèle NormalizedJobOffer
    fieldnames = list(NormalizedJobOffer.model_fields.keys())
    
    # Vérifier si le fichier existe déjà
    file_exists = os.path.exists(output_file)
    
    # Ouvrir le fichier CSV en mode append si il existe, sinon en mode write
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Écrire l'en-tête seulement si le fichier est nouveau
        if not file_exists:
            writer.writeheader()
            logger.info(f"Création du fichier CSV avec les colonnes: {', '.join(fieldnames)}")
        else:
            logger.info(f"Ajout de données au fichier CSV existant: {output_file}")
        
        # Compteur total d'offres
        total_offers = 0
        
        # Boucler sur les catégories Adzuna
        for category in ADZUNA_CATEGORIES:
            logger.info(f"Récupération des offres pour la catégorie {category}")
            
            # Paramètres de pagination
            page = 1
            max_pages = 20  # Limite à 20 pages
            results_per_page = 50  # Nombre d'offres par page
            
            while page <= max_pages:
                try:
                    # Configurer les paramètres de recherche
                    search_params = {
                        "category": category,
                        "results_per_page": results_per_page,
                        "sort_by": SortBy.DATE,
                    }
                    
                    # Récupérer les offres
                    results = await adzuna_client.search_jobs(
                        country=CountryCode.FR,
                        page=page,
                        **search_params
                    )
                    
                    jobs = results.results
                    
                    if not jobs:
                        logger.info(f"Aucune offre trouvée pour la catégorie {category} à la page {page}")
                        break
                    
                    # Normaliser et écrire les offres dans le CSV
                    offers_in_page = 0
                    for job in jobs:
                        try:
                            # Normaliser l'offre
                            normalized_job = normalizer.normalize_adzuna_job(job.model_dump())
                            
                            # Convertir l'offre normalisée en dictionnaire
                            job_dict = normalized_job.model_dump()
                            
                            # Écrire l'offre dans le CSV
                            writer.writerow(job_dict)
                            offers_in_page += 1
                            total_offers += 1
                            
                        except Exception as job_error:
                            logger.error(f"Erreur lors de la normalisation/écriture de l'offre: {job_error}")
                    
                    # Forcer l'écriture sur le disque
                    csvfile.flush()
                    
                    logger.info(f"Récupéré et normalisé {offers_in_page} offres pour la catégorie {category} (page {page})")
                    logger.info(f"Total d'offres récupérées jusqu'à présent: {total_offers}")
                    
                    # Passer à la page suivante
                    page += 1
                    
                    # Pause pour éviter de surcharger l'API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des offres pour la catégorie {category} (page {page}): {e}")
                    break
    
    logger.info(f"Collecte terminée! {total_offers} offres d'emploi normalisées collectées au total.")
    logger.info(f"Données sauvegardées dans {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
