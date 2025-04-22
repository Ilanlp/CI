"""
Collecteur de données d'emploi pour France Travail
Ce script utilise l'API France Travail pour récupérer les offres d'emploi
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
from france_travail import FranceTravailAPI, SearchParams as FranceSearchParams
from jm_normalizer import JobDataNormalizer, NormalizedJobOffer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Liste des codes ROME liés à l'informatique, la data et l'IT
CODES_ROME_IT = [
    # 1. Développement et Programmation
    "M1805",  # Développeur / Développeuse web
    "M1806",  # Développeur / Développeuse multimédia
    "M1802",  # Analyste Concepteur / Conceptrice informatique
    "M1801",  # Analyste d'étude informatique

    # 2. Ingénierie et Architecture
    "M1807",  # Ingénieur / Ingénieure d'étude informatique
    "M1808",  # Ingénieur / Ingénieure systèmes et réseaux informatiques
    "M1809",  # Ingénieur / Ingénieure concepteur / conceptrice informatique
    "M1810",  # Architecte systèmes et réseaux des territoires connectés
    "M1811",  # Architecte cloud
    "M1812",  # Architecte IoT - Internet des Objets

    # 3. Administration et Sécurité
    "M1813",  # Administrateur / Administratrice réseau informatique
    "M1814",  # Administrateur / Administratrice sécurité informatique
    "M1815",  # Ingénieur / Ingénieure sécurité informatique
    "M1816",  # Ingénieur / Ingénieure sécurité web
    "M1817",  # Expert / Experte en cybersécurité
    "M1818",  # Ingénieur / Ingénieure Cybersécurité Datacenter

    # 4. Data et Infrastructure
    "M1819",  # Technicien / Technicienne Datacenter
    "M1820",  # Urbaniste Datacenter
    "M1821",  # Ingénieur / Ingénieure supervision IT Datacenter
    "M1822",  # Délégué / Déléguée à la protection des données - Data Protection Officer

    # 5. Management et Qualité
    "M1823",  # Chef / Cheffe de projet étude et développement informatique
    "M1824",  # Chef / Cheffe de projet maîtrise d'œuvre informatique
    "M1825",  # Qualiticien / Qualiticienne logiciel en informatique
    "M1826",  # Responsable Green IT

    # 6. Réseaux et Télécoms
    "M1827",  # Technicien / Technicienne réseaux informatiques et télécoms
    "M1828",  # Responsable de maintenance réseaux des territoires connectés
    "M1829",  # Ingénieur / Ingénieure systèmes et réseaux des territoires connectés

    # 7. Maintenance et Support
    "M1830",  # Technicien / Technicienne de maintenance en informatique
]

async def main():
    """Fonction principale"""
    # Charger les variables d'environnement depuis un fichier .env si disponible
    load_dotenv()

    # Récupérer les identifiants d'API depuis les variables d'environnement
    france_travail_id = environ.get("FRANCE_TRAVAIL_ID")
    france_travail_key = environ.get("FRANCE_TRAVAIL_KEY")
    adzuna_app_id = environ.get("ADZUNA_APP_ID")
    adzuna_app_key = environ.get("ADZUNA_APP_KEY")

    # Vérifier que les variables d'environnement sont définies
    if not all([france_travail_id, france_travail_key, adzuna_app_id, adzuna_app_key]):
        missing_vars = []
        if not france_travail_id:
            missing_vars.append("FRANCE_TRAVAIL_ID")
        if not france_travail_key:
            missing_vars.append("FRANCE_TRAVAIL_KEY")
        if not adzuna_app_id:
            missing_vars.append("ADZUNA_APP_ID")
        if not adzuna_app_key:
            missing_vars.append("ADZUNA_APP_KEY")

        raise ValueError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}. "
            "Veuillez définir ces variables dans un fichier .env ou dans votre environnement."
        )

    # Créer le dossier de sortie si nécessaire
    output = environ.get("OUTPUT_DIR", "data/")
    path_absolu = Path(__file__).resolve()
    output_dir = f"{path_absolu.parents[2]}/{output}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialiser l'API France Travail et le normalisateur
    france_travail_api = FranceTravailAPI(france_travail_id, france_travail_key)
    normalizer = JobDataNormalizer(
        adzuna_app_id,
        adzuna_app_key,
        france_travail_id,
        france_travail_key
    )

    # Créer le fichier CSV de sortie
    output_file = os.path.join(output_dir, f"france_travail_jobs_{datetime.now().strftime('%Y%m%d')}.csv")
    
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
        
        # Boucler sur les codes ROME IT
        for code_rome in CODES_ROME_IT:
            logger.info(f"Récupération des offres pour le code ROME {code_rome}")
            
            # Paramètres de pagination
            page = 1
            results_per_page = 50  # Nombre d'offres par page
            has_more_results = True
            
            while has_more_results:
                try:
                    # Calculer l'intervalle de résultats pour cette page
                    start_index = (page - 1) * results_per_page
                    end_index = start_index + results_per_page - 1
                    
                    # Configurer les paramètres de recherche
                    search_params = FranceSearchParams(
                        codeROME=code_rome,
                        range=f"{start_index}-{end_index}",  # Intervalle pour cette page
                        sort=1,  # Tri par date décroissant
                    )
                    
                    # Récupérer les offres
                    results = france_travail_api.search_offers(search_params)
                    jobs = results.resultats
                    
                    # Normaliser et écrire les offres dans le CSV
                    offers_in_page = 0
                    for job in jobs:
                        try:
                            # Normaliser l'offre
                            normalized_job = normalizer.normalize_france_travail_job(job.model_dump())
                            
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
                    
                    logger.info(f"Récupéré et normalisé {offers_in_page} offres pour le code ROME {code_rome} (page {page})")
                    logger.info(f"Total d'offres récupérées jusqu'à présent: {total_offers}")
                    
                    # Vérifier s'il y a plus de résultats
                    if len(jobs) < results_per_page:
                        has_more_results = False
                        logger.info(f"Fin de la pagination pour le code ROME {code_rome}")
                    else:
                        page += 1
                        # Pause pour éviter de surcharger l'API
                        time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des offres pour le code ROME {code_rome} (page {page}): {e}")
                    has_more_results = False
    
    logger.info(f"Collecte terminée! {total_offers} offres d'emploi normalisées collectées au total.")
    logger.info(f"Données sauvegardées dans {output_file}")

if __name__ == "__main__":
    asyncio.run(main())