"""
Normalisateur de données d'emploi pour les API Adzuna et France Travail
Ce script utilise les clients API existants pour récupérer les données
et les normaliser dans un format commun en utilisant pandas.
Version 1.2 - Correctifs pour assurer l'intégrité de la sauvegarde des données.
"""

import sys
import asyncio
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import logging
import shutil
import tempfile
from pydantic import BaseModel
from dotenv import load_dotenv
from os import environ
from pathlib import Path
import urllib.parse
import csv
import re
import json

from adzuna import AdzunaClient, CountryCode, SortBy, SearchParams as AdzunaSearchParams
from france_travail import FranceTravailAPI, SearchParams as FranceSearchParams

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Modèle unifié pour les offres d'emploi
class NormalizedJobOffer(BaseModel):
    """Modèle d'offre d'emploi normalisé pour les deux sources"""

    id: str
    source: str  # 'adzuna' ou 'france_travail'
    title: str
    description: Optional[str] = None
    company_name: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date_created: Optional[datetime] = None
    date_updated: Optional[datetime] = None
    contract_type: Optional[str] = None  # CDI, CDD, etc.
    contract_duration: Optional[str] = None
    working_hours: Optional[str] = None  # Temps plein, temps partiel
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None  # mensuel, annuel, horaire
    experience_required: Optional[str] = None
    category: Optional[str] = None
    sector: Optional[str] = None
    application_url: Optional[str] = None
    source_url: Optional[str] = None
    skills: Optional[List[str]] = None
    remote_work: Optional[bool] = None
    is_handicap_accessible: Optional[bool] = None
    code_rome: Optional[str] = None
    langues: Optional[List[str]] = None
    date_extraction: Optional[datetime] = None


class JobDataNormalizer:
    """
    Classe pour normaliser les données d'emploi provenant de différentes API
    """

    def __init__(
        self,
        adzuna_app_id: str,
        adzuna_app_key: str,
        france_travail_id: str,
        france_travail_key: str,
    ):
        """
        Initialise le normalisateur avec les identifiants d'API

        Args:
            adzuna_app_id: ID d'application Adzuna
            adzuna_app_key: Clé API Adzuna
            france_travail_client_id: ID d'identification France Travail
            france_travail_client_secret: Clé secret d'identification France Travail
        """
        self.adzuna_client = AdzunaClient(adzuna_app_id, adzuna_app_key)
        self.france_travail_api = FranceTravailAPI(
            france_travail_id, france_travail_key
        )

        # Mappings pour les types de contrat
        # Ces mappings vont aider à normaliser les données entre les sources
        self.contract_type_mapping = {
            # Adzuna
            "permanent": "CDI",
            "contract": "CDD",
            # Ces valeurs sont déjà conformes pour France Travail
            "CDI": "CDI",
            "CDD": "CDD",
            "MIS": "Mission intérimaire",
            "SAI": "Travail saisonnier",
            "LIB": "Profession libérale",
        }

        self.working_hours_mapping = {
            # Adzuna
            "full_time": "Temps plein",
            "part_time": "Temps partiel",
            # France Travail
            "Temps plein": "Temps plein",
            "Temps partiel": "Temps partiel",
        }

        self.experience_mapping = {
            # France Travail
            "D": "Débutant accepté",
            "S": "Expérience souhaitée",
            "E": "Expérience exigée",
            # Valeurs normalisées (pour Adzuna, à calculer manuellement)
            "Débutant accepté": "Débutant accepté",
            "Expérience souhaitée": "Expérience souhaitée",
            "Expérience exigée": "Expérience exigée",
        }

        self.appellations = self.set_appellations()

    async def close(self):
        """Ferme les connexions des clients API"""
        await self.adzuna_client.close()
        # France Travail API n'a pas de méthode close car il utilise requests et non httpx

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def set_appellations(self):
        try:
            # Utiliser Path pour le chemin de fichier
            appellations_path = Path("./data/RAW_METIERS.csv")
            if not appellations_path.exists():
                logger.warning(
                    f"Fichier d'appellations non trouvé à {appellations_path}"
                )
                return pd.DataFrame(columns=["code_rome", "libelle_search"])

            appellations = pd.read_csv(appellations_path, sep=",")
            return appellations
        except Exception as e:
            logger.error(f"Erreur lors du chargement des appellations: {e}")
            # Retourner un DataFrame vide mais avec les colonnes nécessaires
            return pd.DataFrame(columns=["code_rome", "libelle_search"])

    async def fetch_adzuna_jobs(
        self,
        search_terms: str = None,
        location: str = None,
        max_results: int = 100,
        code_rome: str = None,
        max_days_old: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les offres d'emploi depuis l'API Adzuna

        Args:
            search_terms: Termes de recherche
            location: Lieu (ville, région)
            max_results: Nombre maximum de résultats à récupérer
            code_rome: Code ROME associé à la recherche

        Returns:
            Liste des offres d'emploi brutes d'Adzuna
        """
        logger.info(
            f"Récupération des offres Adzuna pour '{search_terms}' à '{location}'"
        )

        # Utiliser le max_results fourni, avec une limite raisonnable
        effective_max_results = min(max_results, 10000) if max_results > 0 else 10000
        logger.info(f"Limite de résultats définie à {effective_max_results}")

        # Limiter à 50 résultats par page selon la limitation de l'API
        results_per_page = min(50, effective_max_results)
        page = 1
        all_jobs = []

        # Paramètres de recherche
        search_params = {
            "what_or": search_terms,
            "results_per_page": results_per_page,
            "max_days_old": max_days_old,
            "sort_by": SortBy.DATE,
            # Note: Le paramètre sort_dir n'est plus supporté par l'API Adzuna
        }

        # Ajouter la localisation si elle est fournie
        if location:
            search_params["where"] = location

        try:
            # Récupérer la première page
            search_results = await self.adzuna_client.search_jobs(
                country=CountryCode.FR, page=page, **search_params
            )

            all_jobs.extend(search_results.results)
            logger.info(f"Page {page}: récupéré {len(search_results.results)} offres")

            # Récupérer les pages suivantes si nécessaire
            while (
                len(all_jobs) < effective_max_results
                and len(search_results.results) == results_per_page
            ):
                page += 1
                search_results = await self.adzuna_client.search_jobs(
                    country=CountryCode.FR, page=page, **search_params
                )
                all_jobs.extend(search_results.results)
                logger.info(
                    f"Page {page}: récupéré {len(search_results.results)} offres (total: {len(all_jobs)})"
                )

            # Limiter au nombre maximum demandé
            if len(all_jobs) > effective_max_results:
                logger.info(
                    f"Limitation des {len(all_jobs)} offres aux {effective_max_results} premières"
                )
                all_jobs = all_jobs[:effective_max_results]

            logger.info(f"Récupéré {len(all_jobs)} offres d'emploi depuis Adzuna")
            return [job.model_dump() for job in all_jobs]

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des offres Adzuna: {e}")
            # Tenter de gérer des erreurs spécifiques connues
            if not self._handle_adzuna_error(str(e)):
                # Si l'erreur n'a pas été gérée spécifiquement, la loguer simplement
                logger.error(f"Erreur non gérée: {e}")
            return []

    async def fetch_france_travail_jobs(
        self,
        code_rome: str = None,
        location: str = None,
        max_results: int = 3000,
        max_days_old: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les offres d'emploi depuis l'API France Travail
        avec support de pagination

        Args:
            code_rome: Code ROME du métier recherché
            location: Code INSEE de commune ou département
            max_results: Nombre maximum de résultats à récupérer

        Returns:
            Liste des offres d'emploi brutes de France Travail
        """
        logger.info(
            f"Récupération des offres France Travail pour '{code_rome}' à '{location}'"
        )

        if max_days_old:
            maxCreationDate, minCreationDate = self.get_creation_date_iso(max_days_old)
        else:
            maxCreationDate = None
            minCreationDate = None

        # Initialisation des variables pour la pagination
        all_jobs = []
        page_size = 150  # Taille maximale recommandée par appel d'API
        current_index = 0
        more_results = True

        # Définir le nombre maximum d'offres à récupérer
        # Note: L'API limite à 3000 résultats par recherche
        max_results_to_fetch = min(max_results, 3000) if max_results > 0 else 3000
        logger.info(f"Limite de résultats définie à {max_results_to_fetch}")

        try:
            while more_results and len(all_jobs) < max_results_to_fetch:
                # Calcul du range pour cette page
                remaining = max_results_to_fetch - len(all_jobs)
                end_index = current_index + min(page_size, remaining) - 1

                # Configurer les paramètres de recherche pour cette page
                search_params = FranceSearchParams(
                    codeROME=code_rome,
                    minCreationDate=minCreationDate,
                    maxCreationDate=maxCreationDate,
                    range=f"{current_index}-{end_index}",
                    sort=1,  # Tri par date décroissant
                )

                # Ajouter la localisation si elle est fournie
                if location:
                    # Vérifier si c'est un code département (2 chiffres) ou un code INSEE (5 chiffres)
                    if len(location) == 2:
                        search_params.departement = location
                    else:
                        search_params.commune = location

                # Appel à l'API
                results = self.france_travail_api.search_offers(search_params)
                jobs = results.resultats

                # Ajouter les résultats à notre liste
                all_jobs.extend(jobs)

                # Vérifier s'il y a d'autres résultats à récupérer
                if len(jobs) < min(page_size, remaining):
                    # Moins de résultats que demandé = plus de résultats à récupérer
                    more_results = False
                else:
                    # Préparer la prochaine page
                    current_index = end_index + 1

                # Log pour suivre la progression
                logger.info(f"Récupéré {len(jobs)} offres (total: {len(all_jobs)})")

                # Pause entre les requêtes pour éviter de surcharger l'API
                if more_results:
                    await asyncio.sleep(0.5)

            logger.info(
                f"Total: {len(all_jobs)} offres d'emploi récupérées depuis France Travail"
            )

            return [job.model_dump() for job in all_jobs]

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des offres France Travail: {e}"
            )
            return []

    def normalize_adzuna_job(
        self, job: Dict[str, Any], code_rome: str = None
    ) -> NormalizedJobOffer:
        """
        Normalise une offre d'emploi Adzuna vers le format commun

        Args:
            job: Offre d'emploi brute d'Adzuna
            code_rome: Code ROME associé à la recherche

        Returns:
            Offre d'emploi normalisée
        """
        try:
            # Extraire les informations de localisation
            location_name = None
            if job.get("location") and job["location"].get("display_name"):
                location_name = job["location"]["display_name"]

            # Extraire le nom de l'entreprise
            company_name = None
            if job.get("company") and job["company"].get("display_name"):
                company_name = job["company"]["display_name"]

            # Déterminer la catégorie
            category = None
            if job.get("category") and job["category"].get("label"):
                category = job["category"]["label"]

            # Convertir la date au format datetime
            date_created = None
            if job.get("created"):
                try:
                    date_created = datetime.fromisoformat(
                        job["created"].replace("Z", "+00:00")
                    )
                except ValueError:
                    # Essayer un autre format si nécessaire
                    logger.warning(f"Échec de conversion de date: {job['created']}")
                    try:
                        # Format ISO 8601 alternatif
                        from dateutil import parser

                        date_created = parser.parse(job["created"])
                    except:
                        logger.error(f"Impossible de parser la date: {job['created']}")

            # Déterminer le type de contrat normalisé
            contract_type = None
            if job.get("contract_type"):
                contract_type = self.contract_type_mapping.get(
                    job["contract_type"], job["contract_type"]
                )

            # Déterminer le régime horaire normalisé
            working_hours = None
            if job.get("contract_time"):
                working_hours = self.working_hours_mapping.get(
                    job["contract_time"], job["contract_time"]
                )

            # Déterminer l'expérience requise (Adzuna ne fournit pas cela directement)
            experience_required = None

            # Gérer les informations de salaire
            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            salary_currency = "EUR"  # Par défaut pour la France
            salary_period = "annuel"  # Par défaut pour Adzuna

            # date_extraction
            date_extraction = datetime.now()

            # Créer l'offre normalisée
            normalized_job = NormalizedJobOffer(
                id=job["id"],
                source="adzuna",
                title=job["title"],
                description=job.get("description"),
                company_name=company_name,
                location_name=location_name,
                latitude=job.get("latitude"),
                longitude=job.get("longitude"),
                date_created=date_created,
                date_updated=date_created,  # Adzuna ne fournit pas de date de mise à jour
                contract_type=contract_type,
                contract_duration=None,  # Adzuna ne fournit pas cette information
                working_hours=working_hours,
                salary_min=salary_min,
                salary_max=salary_max,
                salary_currency=salary_currency,
                salary_period=salary_period,
                experience_required=experience_required,
                category=category,
                sector=None,  # Adzuna ne fournit pas cette information clairement
                application_url=job.get("redirect_url"),
                source_url=job.get("redirect_url"),
                skills=None,  # Adzuna ne fournit pas cette information
                remote_work=None,  # Adzuna ne fournit pas cette information
                is_handicap_accessible=None,
                code_rome=code_rome,
                langues=None,
                date_extraction=date_extraction,
            )

            return normalized_job
        except Exception as e:
            logger.error(f"Erreur lors de la normalisation d'une offre Adzuna: {e}")
            # Enregistrer l'offre problématique pour diagnostic
            try:
                with open("error_job_adzuna.json", "w") as f:
                    json.dump(job, f, default=str)
                logger.info(
                    "Offre problématique sauvegardée dans error_job_adzuna.json"
                )
            except:
                pass
            # Re-lever l'exception pour être gérée au niveau supérieur
            raise

    def normalize_france_travail_job(self, job: Dict[str, Any]) -> NormalizedJobOffer:
        """
        Normalise une offre d'emploi France Travail vers le format commun

        Args:
            job: Offre d'emploi brute de France Travail

        Returns:
            Offre d'emploi normalisée
        """
        try:
            # Extraire les informations de localisation
            location_name = None
            latitude = None
            longitude = None

            if job.get("lieuTravail"):
                location_name = job["lieuTravail"].get("libelle")
                latitude = job["lieuTravail"].get("latitude")
                longitude = job["lieuTravail"].get("longitude")

            # Extraire le nom de l'entreprise
            company_name = None
            if job.get("entreprise"):
                company_name = job["entreprise"].get("nom")

            # Déterminer la catégorie et le secteur
            category = job.get("romeLibelle")
            sector = job.get("secteurActiviteLibelle")

            # Convertir les dates au format datetime
            # date_created = job.get("dateCreation")
            # date_updated = job.get("dateActualisation")
            date_created = datetime.fromisoformat(
                str(job.get("dateCreation")).replace("Z", "+00:00")
            )
            date_updated = datetime.fromisoformat(
                str(job.get("dateActualisation")).replace("Z", "+00:00")
            )

            # Déterminer le type de contrat normalisé
            contract_type = None
            if job.get("typeContrat"):
                contract_type = self.contract_type_mapping.get(
                    job["typeContrat"], job["typeContrat"]
                )

            # Extraire la durée du contrat (pour les CDD)
            contract_duration = job.get("typeContratLibelle")
            if contract_duration == contract_type:
                contract_duration = None

            # Déterminer le régime horaire normalisé
            working_hours = None
            if job.get("dureeTravailLibelleConverti"):
                working_hours = self.working_hours_mapping.get(
                    job["dureeTravailLibelleConverti"],
                    job["dureeTravailLibelleConverti"],
                )

            # Déterminer l'expérience requise
            experience_required = None
            if job.get("experienceExige"):
                experience_required = self.experience_mapping.get(
                    job["experienceExige"], job.get("experienceLibelle")
                )

            # Gérer les informations de salaire
            # France Travail ne donne pas directement min/max, on doit parser le libellé
            salary_min = None
            salary_max = None
            salary_currency = "EUR"  # Par défaut pour la France
            salary_period = None

            if job.get("salaire") and job["salaire"].get("libelle"):
                salary_info = job["salaire"]["libelle"]

                # Déterminer la période de salaire
                if "Mensuel" in salary_info:
                    salary_period = "mensuel"
                elif "Annuel" in salary_info:
                    salary_period = "annuel"
                elif "Horaire" in salary_info:
                    salary_period = "horaire"

                # Tenter d'extraire les valeurs de salaire avec regex
                try:
                    # Rechercher des patterns comme "de 1500.00 Euros à 2000.00 Euros"
                    import re

                    # Pattern pour "de X à Y"
                    range_pattern = r"de (\d+[.,]?\d*) .+ à (\d+[.,]?\d*)"
                    range_match = re.search(range_pattern, salary_info)

                    if range_match:
                        salary_min = float(range_match.group(1).replace(",", "."))
                        salary_max = float(range_match.group(2).replace(",", "."))
                    else:
                        # Pattern pour "de X" ou "X Euros"
                        single_pattern = r"(?:de )?(\d+[.,]?\d*) (?:Euro|€)"
                        single_match = re.search(single_pattern, salary_info)
                        if single_match:
                            val = float(single_match.group(1).replace(",", "."))
                            salary_min = salary_max = val
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Erreur lors du parsing du salaire: {salary_info}, {str(e)}"
                    )
                    # En cas d'échec du parsing, laisser à None

            # Extraire les compétences
            skills = []
            if job.get("competences"):
                skills = [
                    comp["libelle"] for comp in job["competences"] if "libelle" in comp
                ]

            # Déterminer l'URL de candidature
            application_url = None
            source_url = None

            if job.get("contact") and job["contact"].get("urlPostulation"):
                application_url = str(job["contact"]["urlPostulation"])

            if job.get("origineOffre") and job["origineOffre"].get("urlOrigine"):
                source_url = str(job["origineOffre"]["urlOrigine"])

            # Déterminer l'accessibilité aux travailleurs handicapés
            is_handicap_accessible = job.get("accessibleTH")

            # Extraire le code ROME
            code_rome = job.get("romeCode")

            # Extraire les langues
            langues = []
            if job.get("langues"):
                langues = [
                    comp["libelle"] for comp in job["langues"] if "libelle" in comp
                ]
            # date_extraction
            date_extraction = datetime.now()

            # Créer l'offre normalisée
            normalized_job = NormalizedJobOffer(
                id=job["id"],
                source="france_travail",
                title=job["intitule"],
                description=job.get("description"),
                company_name=company_name,
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                date_created=date_created,
                date_updated=date_updated,
                contract_type=contract_type,
                contract_duration=contract_duration,
                working_hours=working_hours,
                salary_min=salary_min,
                salary_max=salary_max,
                salary_currency=salary_currency,
                salary_period=salary_period,
                experience_required=experience_required,
                category=category,
                sector=sector,
                application_url=application_url,
                source_url=source_url,
                skills=skills if skills else None,
                remote_work=None,  # Il faudrait analyser le texte pour déterminer cela
                is_handicap_accessible=is_handicap_accessible,
                code_rome=code_rome,
                langues=langues if langues else None,
                date_extraction=date_extraction,
            )

            return normalized_job
        except Exception as e:
            logger.error(
                f"Erreur lors de la normalisation d'une offre France Travail: {e}"
            )
            # Enregistrer l'offre problématique pour diagnostic
            try:
                with open("error_job_france_travail.json", "w") as f:
                    json.dump(job, f, default=str)
                logger.info(
                    "Offre problématique sauvegardée dans error_job_france_travail.json"
                )
            except:
                pass
            # Re-lever l'exception pour être gérée au niveau supérieur
            raise

    async def fetch_and_normalize_adzuna(
        self,
        code_rome: str = None,
        location: str = None,
        max_results: int = 100,
        max_days_old: int = None,
    ) -> List[NormalizedJobOffer]:
        """
        Récupère et normalise les offres d'emploi Adzuna de manière optimisée
        en limitant les appels API à 1 par seconde.

        Args:
            code_rome: Code(s) ROME séparés par des virgules
            location: Lieu (ville, région)
            max_results: Nombre maximum de résultats par code ROME

        Returns:
            Liste des offres d'emploi normalisées
        """
        # S'assurer que les appellations sont chargées
        if (
            not hasattr(self, "appellations")
            or self.appellations is None
            or self.appellations.empty
        ):
            self.set_appellations()

        # Si appellations est toujours vide, c'est un problème critique
        if self.appellations.empty:
            logger.error(
                "Impossible de charger les appellations. Vérifiez le fichier RAW_METIERS.csv"
            )
            return []

        # Préparation des paramètres d'appel API
        search_params = []
        all_codes = [code.strip() for code in code_rome.split(",")]
        logger.info(f"Recherche pour les codes ROME: {all_codes}")

        # Préparation des tâches d'appel API et calcul des max_results ajustés
        for code in all_codes:
            df_filtered = self.appellations[self.appellations["code_rome"] == code]
            if df_filtered.empty:
                logger.warning(f"Aucune appellation trouvée pour le code ROME {code}")
                continue

            search_terms = df_filtered["libelle_search"].tolist()
            logger.info(f"Code ROME {code}: {len(search_terms)} termes de recherche")

            # Calculer le nombre ajusté de résultats par recherche
            # Si nous avons plusieurs termes de recherche, diviser max_results
            adjusted_max = (
                max(1, max_results // len(search_terms)) if search_terms else 0
            )

            if adjusted_max == 0:
                logger.warning(
                    f"Max_results ajusté à 0 pour {code}, aucune recherche effectuée"
                )
                continue

            # Ajouter chaque terme de recherche à la liste des paramètres
            joined_terms = " ".join(search_terms)
            search_params.append((joined_terms, location, adjusted_max, code, max_days_old))

        if not search_params:
            logger.warning(
                "Aucun paramètre de recherche valide. Vérifiez les codes ROME."
            )
            return []

        # Initialiser un sémaphore pour limiter les appels concurrents
        # Nous utilisons un sémaphore de 1 pour ne permettre qu'un seul appel à la fois
        api_semaphore = asyncio.Semaphore(1)

        # Fonction qui encapsule l'appel à l'API avec limitation de taux
        async def rate_limited_api_call(
            search, location, max_results, code, max_days_old
        ):
            async with api_semaphore:
                # Effectuer l'appel API
                result = await self.fetch_adzuna_jobs(
                    search, location, max_results, code, max_days_old
                )
                # Attendre 1 seconde avant de libérer le sémaphore pour limiter à 1 appel/seconde
                await asyncio.sleep(1)
                return result

        # Créer toutes les tâches asynchrones avec limitation de taux
        tasks = [rate_limited_api_call(*params) for params in search_params]

        # Exécuter toutes les tâches et rassembler les résultats
        # Nous utilisons asyncio.gather pour exécuter toutes les tâches en parallèle
        # mais le sémaphore s'assurera qu'une seule tâche pourra appeler l'API à la fois
        raw_results = await asyncio.gather(*tasks)

        # Compter le nombre total d'offres brutes récupérées
        total_raw_jobs = sum(len(raw_jobs) for raw_jobs in raw_results)
        logger.info(f"Total des offres brutes récupérées: {total_raw_jobs}")

        # Traiter les résultats et normaliser les offres d'emploi avec gestion d'erreurs
        normalized_jobs = []
        errors_count = 0

        for i, raw_jobs in enumerate(raw_results):
            # unpack search_params
            logger.info(
                f"unpack search_params {search_params[i]} : {len(search_params[i])}"
            )
            _, _, _, code, _ = search_params[i]

            # Normaliser chaque offre avec gestion d'erreurs
            for job in raw_jobs:
                try:
                    normalized_job = self.normalize_adzuna_job(job, code)
                    normalized_jobs.append(normalized_job)
                except Exception as e:
                    errors_count += 1
                    logger.error(f"Erreur lors de la normalisation d'une offre: {e}")
                    continue

        logger.info(
            f"Normalisation: {len(normalized_jobs)} offres traitées, {errors_count} erreurs"
        )

        # Déduplication des offres par ID
        before_dedup = len(normalized_jobs)
        normalized_jobs_dedup = {}

        for job in normalized_jobs:
            # Utiliser l'ID combiné avec la source comme clé de déduplication
            dedup_key = f"{job.source}_{job.id}"
            normalized_jobs_dedup[dedup_key] = job

        normalized_jobs = list(normalized_jobs_dedup.values())
        after_dedup = len(normalized_jobs)

        logger.info(f"Déduplication: {before_dedup - after_dedup} doublons supprimés")

        # Limiter le nombre total de résultats si nécessaire
        if len(normalized_jobs) > max_results:
            logger.info(f"Limitation à {max_results} offres sur {len(normalized_jobs)}")
            normalized_jobs = normalized_jobs[:max_results]

        logger.info(f"Normalisé {len(normalized_jobs)} offres Adzuna au total")
        return normalized_jobs

    async def fetch_and_normalize_france_travail(
        self,
        code_rome: str = None,
        location: str = None,
        max_results: int = 3000,
        max_days_old: int = None,
    ) -> List[NormalizedJobOffer]:
        """
        Récupère et normalise les offres d'emploi France Travail

        Args:
            code_rome: Code ROME du métier recherché
            location: Code INSEE de commune ou département
            max_results: Nombre maximum de résultats

        Returns:
            Liste des offres d'emploi normalisées
        """
        # Vérifier les paramètres
        if not code_rome:
            logger.warning("Aucun code ROME fourni pour la recherche France Travail")
            return []

        # Diviser les codes ROME s'il y en a plusieurs
        codes_rome = [code.strip() for code in code_rome.split(",")]
        all_normalized_jobs = []

        # Traiter chaque code ROME séparément
        for code in codes_rome:
            logger.info(f"Recherche France Travail pour code ROME: {code}")

            # Utiliser la méthode de récupération avec pagination
            raw_jobs = await self.fetch_france_travail_jobs(
                code, location, max_results // len(codes_rome), max_days_old
            )

            logger.info(
                f"Récupéré {len(raw_jobs)} offres France Travail pour le code {code}"
            )

            # Normaliser chaque offre avec gestion d'erreurs
            normalized_jobs = []
            errors_count = 0

            for job in raw_jobs:
                try:
                    normalized_job = self.normalize_france_travail_job(job)
                    normalized_jobs.append(normalized_job)
                except Exception as e:
                    errors_count += 1
                    logger.error(
                        f"Erreur lors de la normalisation d'une offre France Travail: {e}"
                    )
                    continue

            logger.info(
                f"Normalisation France Travail: {len(normalized_jobs)} offres traitées, {errors_count} erreurs"
            )
            all_normalized_jobs.extend(normalized_jobs)

        # Déduplication des offres par ID
        before_dedup = len(all_normalized_jobs)
        normalized_jobs_dedup = {}

        for job in all_normalized_jobs:
            # Utiliser l'ID combiné avec la source comme clé de déduplication
            dedup_key = f"{job.source}_{job.id}"
            normalized_jobs_dedup[dedup_key] = job

        all_normalized_jobs = list(normalized_jobs_dedup.values())
        after_dedup = len(all_normalized_jobs)

        logger.info(
            f"Déduplication France Travail: {before_dedup - after_dedup} doublons supprimés"
        )

        # Limiter au nombre maximum demandé
        if len(all_normalized_jobs) > max_results:
            logger.info(
                f"Limitation à {max_results} offres France Travail sur {len(all_normalized_jobs)}"
            )
            all_normalized_jobs = all_normalized_jobs[:max_results]

        return all_normalized_jobs

    async def fetch_and_normalize_all(
        self,
        code_rome: str = None,
        location_adzuna: str = None,
        location_france_travail: str = None,
        max_results: int = 100,
        max_days_old: int = None,
    ) -> List[NormalizedJobOffer]:
        """
        Récupère et normalise les offres d'emploi des deux sources

        Args:
            code_rome: Code(s) ROME séparés par des virgules
            location_adzuna: Lieu pour Adzuna (ville, région)
            location_france_travail: Code INSEE/département pour France Travail
            max_results: Nombre maximum de résultats par source

        Returns:
            Liste combinée des offres d'emploi normalisées
        """
        if not code_rome:
            logger.error("Aucun code ROME fourni. Impossible de continuer.")
            return []

        logger.info(
            f"Récupération des offres pour {code_rome} (max {max_results} par source)"
        )

        # Exécuter les deux récupérations en parallèle pour gagner du temps
        adzuna_task = self.fetch_and_normalize_adzuna(
            code_rome, location_adzuna, max_results, max_days_old
        )

        france_travail_task = self.fetch_and_normalize_france_travail(
            code_rome, location_france_travail, max_results, max_days_old
        )

        # Attendre les deux tâches
        adzuna_jobs, france_travail_jobs = await asyncio.gather(
            adzuna_task, france_travail_task
        )

        logger.info(
            f"Récupéré {len(adzuna_jobs)} offres Adzuna et {len(france_travail_jobs)} offres France Travail"
        )

        # Combiner les résultats
        all_jobs = adzuna_jobs + france_travail_jobs

        # Déduplication finale au cas où (bien que chaque source ait déjà dédupliqué)
        before_dedup = len(all_jobs)
        normalized_jobs_dedup = {}

        for job in all_jobs:
            # Utiliser l'ID combiné avec la source comme clé de déduplication
            dedup_key = f"{job.source}_{job.id}"
            normalized_jobs_dedup[dedup_key] = job

        all_jobs = list(normalized_jobs_dedup.values())
        after_dedup = len(all_jobs)

        if before_dedup > after_dedup:
            logger.info(
                f"Déduplication finale: {before_dedup - after_dedup} doublons supprimés"
            )

        logger.info(f"Total final: {len(all_jobs)} offres d'emploi normalisées")
        return all_jobs

    def _sanitize_text(self, text):
        """
        Nettoie le texte pour éviter les problèmes dans les CSV,
        tout en préservant les caractères importants

        Args:
            text: Texte à nettoyer

        Returns:
            Texte nettoyé
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # Préserver les sauts de ligne mais les remplacer par des espaces
        # pour éviter les problèmes avec CSV tout en gardant la lisibilité
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

        # Échapper les guillemets pour le CSV (doublement des guillemets)
        text = text.replace('"', '""')

        # Remplacer les caractères de contrôle invisibles par des espaces
        # au lieu de les supprimer complètement
        # Mais préserver les tabulations, espaces, etc.
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", " ", text)

        # Remplacer les séquences de plusieurs espaces par un seul espace
        text = re.sub(r"\s+", " ", text)

        # Retirer les espaces au début et à la fin
        text = text.strip()

        return text

    def create_dataframe(
        self, normalized_jobs: List[NormalizedJobOffer]
    ) -> pd.DataFrame:
        """
        Crée un DataFrame pandas à partir des offres normalisées
        avec vérification d'intégrité et gestion des types de données

        Args:
            normalized_jobs: Liste des offres d'emploi normalisées

        Returns:
            DataFrame pandas
        """
        if not normalized_jobs:
            logger.warning("Aucune offre à convertir en DataFrame")
            # Retourner un DataFrame vide mais avec les colonnes correctes
            return pd.DataFrame(
                columns=[
                    "id",
                    "source",
                    "title",
                    "description",
                    "company_name",
                    "location_name",
                    "latitude",
                    "longitude",
                    "date_created",
                    "date_updated",
                    "contract_type",
                    "contract_duration",
                    "working_hours",
                    "salary_min",
                    "salary_max",
                    "salary_currency",
                    "salary_period",
                    "experience_required",
                    "category",
                    "sector",
                    "application_url",
                    "source_url",
                    "skills",
                    "remote_work",
                    "is_handicap_accessible",
                    "code_rome",
                    "langues",
                    "date_extraction",
                ]
            )

        logger.info(f"Conversion de {len(normalized_jobs)} offres en DataFrame")

        # Convertir les offres en dictionnaires avec gestion d'erreurs
        jobs_dicts = []
        for job in normalized_jobs:
            try:
                jobs_dicts.append(job.model_dump())
            except Exception as e:
                logger.error(f"Erreur lors de la conversion d'une offre en dict: {e}")
                continue

        # Créer un DataFrame
        df = pd.DataFrame(jobs_dicts)

        # Vérifier la cohérence des données
        logger.info(
            f"DataFrame créé avec {len(df)} lignes et {len(df.columns)} colonnes"
        )

        # Conversion des colonnes de date en datetime (au cas où)
        for date_col in ["date_created", "date_updated", "date_extraction"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                missing_dates = df[date_col].isna().sum()
                if missing_dates > 0:
                    logger.warning(
                        f"{missing_dates} valeurs de date manquantes dans {date_col}"
                    )

        # Sanitize les données textuelles pour éviter les problèmes CSV
        text_columns = [
            "title",
            "description",
            "company_name",
            "location_name",
            "contract_type",
            "contract_duration",
            "working_hours",
            "category",
            "sector",
        ]

        for col in text_columns:
            if col in df.columns:
                # Remplacer les caractères problématiques et les valeurs nulles
                df[col] = df[col].apply(
                    lambda x: self._sanitize_text(x) if x is not None else ""
                )

        # Traiter les listes (comme les compétences et langues)
        for list_col in ["skills", "langues"]:
            if list_col in df.columns:
                df[list_col] = df[list_col].apply(
                    lambda x: (
                        "; ".join(self._sanitize_text(s) for s in x if s)
                        if isinstance(x, list)
                        else ""
                    )
                )

        # S'assurer que les valeurs numériques sont correctes
        numeric_columns = ["salary_min", "salary_max", "latitude", "longitude"]
        for col in numeric_columns:
            if col in df.columns:
                # Conversion en numérique avec remplacement des erreurs par NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Vérifier les valeurs aberrantes pour les salaires
                if col in ["salary_min", "salary_max"] and not df[col].isna().all():
                    # Vérifier s'il y a des valeurs négatives
                    neg_values = (df[col] < 0).sum()
                    if neg_values > 0:
                        logger.warning(f"{neg_values} valeurs négatives dans {col}")

                    # Vérifier les valeurs extrêmes (>1M pour les salaires)
                    extreme_values = (df[col] > 1000000).sum()
                    if extreme_values > 0:
                        logger.warning(f"{extreme_values} valeurs extrêmes dans {col}")

        # Remplir les valeurs manquantes avec des valeurs appropriées
        df = df.fillna(
            {
                "description": "",
                "company_name": "Non spécifié",
                "location_name": "Non spécifié",
                "contract_type": "Non spécifié",
                "working_hours": "Non spécifié",
                "experience_required": "Non spécifié",
                "category": "Non spécifié",
                "sector": "Non spécifié",
                "salary_currency": "",
                "salary_period": "",
                "application_url": "",
                "source_url": "",
            }
        )

        # Pour les colonnes numériques, ne pas les remplacer par une chaîne vide
        # mais laisser le NaN pour une meilleure compatibilité statistique

        # Vérifier les duplications d'ID dans le DataFrame
        if "id" in df.columns and "source" in df.columns:
            duplicate_keys = df.duplicated(subset=["id", "source"]).sum()
            if duplicate_keys > 0:
                logger.warning(
                    f"Trouvé {duplicate_keys} offres en double dans le DataFrame"
                )
                # Supprimer les doublons
                df = df.drop_duplicates(subset=["id", "source"])
                logger.info(f"DataFrame après déduplication: {len(df)} lignes")

        return df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        chunk_size: int = 10000,
        compress: bool = False,
    ) -> str:
        """
        Sauvegarde le DataFrame dans un fichier CSV, avec support pour les grands volumes
        de données et vérification d'intégrité

        Args:
            df: DataFrame à sauvegarder
            filename: Nom du fichier (sans extension)
            chunk_size: Taille des chunks pour la sauvegarde en mode chunksize
            compress: Si True, compresse le fichier en gzip

        Returns:
            Chemin du fichier sauvegardé ou None en cas d'erreur
        """
        if df.empty:
            logger.warning("Tentative de sauvegarde d'un DataFrame vide")
            return None

        # Nettoyer le nom de fichier
        safe_filename = self._sanitize_filename(filename)

        # Déterminer l'extension et les paramètres de compression
        extension = ".csv.gz" if compress else ".csv"
        filepath = Path(f"{safe_filename}{extension}")
        compression = "gzip" if compress else None

        # Créer le répertoire parent si nécessaire
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Fichier temporaire pour écriture sécurisée
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            temp_filepath = tmp_file.name
            logger.info(f"Sauvegarde temporaire dans {temp_filepath}")

            try:
                # Nombre total de lignes à sauvegarder
                total_rows = len(df)
                logger.info(f"Sauvegarde de {total_rows} lignes de données")

                # Si le DataFrame est trop grand pour une sauvegarde directe
                if total_rows > chunk_size:
                    logger.info(f"Sauvegarde d'un grand DataFrame en mode chunk")

                    # Compter les lignes écrites pour vérification d'intégrité
                    rows_written = 0

                    # Écrire l'en-tête d'abord
                    df.iloc[0:0].to_csv(
                        temp_filepath,
                        index=False,
                        encoding="utf-8",
                        quoting=csv.QUOTE_ALL,
                        escapechar="\\",
                        compression=compression,
                    )

                    # Écrire les données par chunks
                    for chunk_start in range(0, total_rows, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_rows)
                        chunk_size_actual = chunk_end - chunk_start
                        logger.info(
                            f"Écriture du chunk {chunk_start+1}-{chunk_end} sur {total_rows}"
                        )

                        # Écrire le chunk
                        df.iloc[chunk_start:chunk_end].to_csv(
                            temp_filepath,
                            index=False,
                            encoding="utf-8",
                            quoting=csv.QUOTE_ALL,
                            escapechar="\\",
                            mode="a",
                            header=False,
                            compression=compression,
                        )

                        rows_written += chunk_size_actual
                        logger.info(
                            f"Progression: {rows_written}/{total_rows} lignes écrites"
                        )

                else:
                    # Pour les petits DataFrames, utiliser la méthode standard
                    df.to_csv(
                        temp_filepath,
                        index=False,
                        encoding="utf-8",
                        quoting=csv.QUOTE_ALL,
                        escapechar="\\",
                        compression=compression,
                    )
                    rows_written = total_rows

                # Vérification d'intégrité
                logger.info(
                    f"Vérification d'intégrité: {rows_written} lignes écrites sur {total_rows} attendues"
                )
                if rows_written != total_rows:
                    logger.error(
                        f"ERREUR D'INTÉGRITÉ: Nombre de lignes écrites incorrect"
                    )
                    raise IOError(
                        f"Intégrité compromise: {rows_written} lignes écrites sur {total_rows}"
                    )

                # Déplacer le fichier temporaire vers la destination finale de manière atomique
                try:
                    # Backup du fichier existant si nécessaire
                    if filepath.exists():
                        backup_path = filepath.with_suffix(filepath.suffix + ".bak")
                        logger.info(
                            f"Sauvegarde du fichier existant vers {backup_path}"
                        )
                        filepath.rename(backup_path)

                    # Déplacer le fichier temporaire
                    shutil.move(temp_filepath, filepath)
                    logger.info(f"Fichier déplacé avec succès vers {filepath}")

                except Exception as move_error:
                    logger.error(f"Erreur lors du déplacement du fichier: {move_error}")
                    raise

                # Vérification finale du fichier
                if not filepath.exists():
                    logger.error("ERREUR CRITIQUE: Le fichier final n'existe pas")
                    raise FileNotFoundError(
                        f"Le fichier {filepath} n'a pas été créé correctement"
                    )

                # Vérifier la taille du fichier
                file_size = filepath.stat().st_size
                logger.info(f"Taille du fichier final: {file_size/1024/1024:.2f} MB")

                if file_size == 0:
                    logger.error("ERREUR CRITIQUE: Le fichier final est vide")
                    raise IOError(f"Le fichier {filepath} est vide")

                logger.info(f"Données sauvegardées avec succès dans {filepath}")
                return str(filepath)

            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du fichier CSV: {e}")

                # Supprimer le fichier temporaire en cas d'erreur
                try:
                    if Path(temp_filepath).exists():
                        Path(temp_filepath).unlink()
                        logger.info(f"Fichier temporaire supprimé: {temp_filepath}")
                except:
                    pass

                # Tenter une sauvegarde avec des options plus sécurisées si ce n'était pas déjà le cas
                if not compress:
                    logger.info("Tentative de sauvegarde alternative...")
                    try:
                        alternative_filepath = Path(f"{safe_filename}_alternative.csv")

                        # Utiliser un encodage différent et des paramètres plus sécurisés
                        df.to_csv(
                            alternative_filepath,
                            index=False,
                            encoding="utf-8-sig",  # UTF-8 avec BOM pour meilleure compatibilité
                            quoting=csv.QUOTE_ALL,
                            escapechar="\\",
                            line_terminator="\n",
                        )
                        logger.info(
                            f"Données sauvegardées (méthode alternative) dans {alternative_filepath}"
                        )
                        return str(alternative_filepath)
                    except Exception as e2:
                        logger.error(f"Échec de la sauvegarde alternative: {e2}")

                # Si toutes les tentatives échouent, lever une exception
                raise IOError(f"Impossible de sauvegarder les données: {str(e)}")

    def _sanitize_filename(self, filename):
        """
        Nettoie le nom de fichier pour éviter les problèmes de sécurité

        Args:
            filename: Nom de fichier à nettoyer

        Returns:
            Nom de fichier sécurisé
        """
        # Convertir en Path pour manipuler correctement
        filepath = Path(filename)

        # Extraire le nom de base (sans le chemin)
        basename = filepath.name

        # Remplacer les caractères dangereux
        import re

        safe_name = re.sub(r'[\\/*?:"<>|]', "_", basename)

        # Limiter la longueur
        if len(safe_name) > 200:
            stem = safe_name[:196]  # Laisser de la place pour l'extension
            safe_name = stem + safe_name[-4:] if "." in safe_name[-5:] else stem

        # Reconstruire le chemin
        return str(filepath.parent / safe_name)

    def _handle_adzuna_error(self, error_message: str) -> bool:
        """
        Traite les erreurs spécifiques d'Adzuna et fournit une information utile

        Args:
            error_message: Message d'erreur original

        Returns:
            bool: True si l'erreur a été gérée, False sinon
        """
        # Vérifier si l'erreur est liée à un paramètre non supporté
        if "sort_dir" in error_message.lower():
            logger.warning(
                "Le paramètre 'sort_dir' n'est plus supporté par l'API Adzuna. L'erreur a été gérée."
            )
            return True

        # Erreur liée à une limite d'API dépassée
        if "rate limit" in error_message.lower() or "429" in error_message:
            logger.warning(
                "Limite de taux d'API Adzuna atteinte. Attendez quelques minutes avant de réessayer."
            )
            return True

        # Erreur d'authentification
        if "401" in error_message or "unauthorized" in error_message.lower():
            logger.error(
                "Erreur d'authentification Adzuna. Vérifiez vos identifiants API."
            )
            return True

        # Erreur de paramètres invalides
        if "400" in error_message or "bad request" in error_message.lower():
            logger.error(
                "Requête Adzuna invalide. Vérifiez les paramètres de recherche."
            )
            return True

        # Autres cas d'erreurs spécifiques peuvent être ajoutés ici
        return False

    def url_encode_params(self, params):
        """
        Encode les paramètres de requête URL.

        Args:
            params (dict): Dictionnaire contenant les paramètres à encoder
                        Les valeurs peuvent être des strings, nombres, booléens, listes ou None

        Returns:
            str: Chaîne encodée au format query string
        """

        # Pour traiter les listes et les valeurs spéciales
        def process_value(value):
            if value is None:
                return ""
            elif isinstance(value, list) or isinstance(value, tuple):
                # Pour les listes, on utilise la notation param=val1&param=val2
                return value
            elif isinstance(value, bool):
                return "true" if value else "false"
            else:
                return str(value)

        # Préparer les paramètres en traitant les types spéciaux
        processed_params = {}
        for key, value in params.items():
            processed_params[key] = process_value(value)

        # Encoder les paramètres
        encoded_params = urllib.parse.urlencode(processed_params, doseq=True)

        return encoded_params

    def get_creation_date_iso(self, max_days_old: int = 1):
        """
        Retourne les dates d'aujourd'hui et d'hier au format ISO 8601

        Returns:
            tuple: (today_str, yesterday_str) au format ISO 8601
        """
        now_utc = datetime.now(timezone.utc)
        yesterday_utc = now_utc - timedelta(days=max_days_old)

        # Format ISO 8601 avec le suffixe Z (UTC)
        today_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        yesterday_str = yesterday_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        return today_str, yesterday_str

    def process_and_analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Traite et analyse les données du DataFrame

        Args:
            df: DataFrame des offres d'emploi

        Returns:
            Dictionnaire des analyses
        """
        if df.empty:
            logger.warning("DataFrame vide, analyse impossible")
            return {
                "total_offers": 0,
                "sources": {},
                "contracts": {},
                "salary_stats": {},
                "categories": {},
                "experience": {},
            }

        # Copier le DataFrame pour éviter de modifier l'original
        df_analysis = df.copy()

        # Enregistrer des statistiques de base
        logger.info(f"Analyse de {len(df_analysis)} offres d'emploi")

        # Nombre d'offres par source
        source_counts = df_analysis["source"].value_counts().to_dict()
        logger.info(f"Répartition par source: {source_counts}")

        # Types de contrat les plus courants
        contract_counts = df_analysis["contract_type"].value_counts().head(5).to_dict()
        logger.info(f"Types de contrat les plus courants: {contract_counts}")

        # Statistiques de salaire
        salary_stats = {}

        # Vérifier si les colonnes existent et ne sont pas toutes NA
        if (
            "salary_min" in df_analysis.columns
            and df_analysis["salary_min"].notna().any()
        ):
            # Statistiques pour les salaires minimum
            salary_stats["min"] = {
                "mean": float(df_analysis["salary_min"].mean()),
                "median": float(df_analysis["salary_min"].median()),
                "min": float(df_analysis["salary_min"].min()),
                "max": float(df_analysis["salary_min"].max()),
                "count": int(df_analysis["salary_min"].count()),
                "percent_filled": float(df_analysis["salary_min"].notna().mean() * 100),
            }
            logger.info(f"Statistiques salary_min: {salary_stats['min']}")

        if (
            "salary_max" in df_analysis.columns
            and df_analysis["salary_max"].notna().any()
        ):
            # Statistiques pour les salaires maximum
            salary_stats["max"] = {
                "mean": float(df_analysis["salary_max"].mean()),
                "median": float(df_analysis["salary_max"].median()),
                "min": float(df_analysis["salary_max"].min()),
                "max": float(df_analysis["salary_max"].max()),
                "count": int(df_analysis["salary_max"].count()),
                "percent_filled": float(df_analysis["salary_max"].notna().mean() * 100),
            }
            logger.info(f"Statistiques salary_max: {salary_stats['max']}")

        # Répartition des offres par catégorie
        category_counts = {}
        if "category" in df_analysis.columns:
            category_counts = df_analysis["category"].value_counts().head(10).to_dict()
            logger.info(f"Top 10 des catégories: {category_counts}")

        # Analyse des expériences requises
        experience_counts = {}
        if "experience_required" in df_analysis.columns:
            experience_counts = (
                df_analysis["experience_required"].value_counts().to_dict()
            )
            logger.info(f"Répartition par expérience: {experience_counts}")

        # Analyse des codes ROME
        rome_counts = {}
        if (
            "code_rome" in df_analysis.columns
            and df_analysis["code_rome"].notna().any()
        ):
            rome_counts = df_analysis["code_rome"].value_counts().head(10).to_dict()
            logger.info(f"Top 10 des codes ROME: {rome_counts}")

        # Résultats de l'analyse
        analysis = {
            "total_offers": len(df_analysis),
            "sources": source_counts,
            "contracts": contract_counts,
            "salary_stats": salary_stats,
            "categories": category_counts,
            "experience": experience_counts,
            "rome_codes": rome_counts,
            "date_analysis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return analysis


async def main():
    """Fonction principale avec gestion d'erreurs robuste"""
    try:
        # Charger les variables d'environnement depuis un fichier .env si disponible
        load_dotenv()

        # Récupérer les identifiants d'API depuis les variables d'environnement
        adzuna_app_id = environ.get("ADZUNA_APP_ID")
        adzuna_app_key = environ.get("ADZUNA_APP_KEY")
        france_travail_id = environ.get("FRANCE_TRAVAIL_ID")
        france_travail_key = environ.get("FRANCE_TRAVAIL_KEY")

        # Vérifier que les variables d'environnement sont définies
        if not all(
            [adzuna_app_id, adzuna_app_key, france_travail_id, france_travail_key]
        ):
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

        # Créer le dossier de sortie avec Path pour une meilleure gestion des chemins
        output = environ.get("OUTPUT_DIR", "data/")
        path_absolu = Path(__file__).resolve()
        output_dir = path_absolu.parent / output
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dossier de sortie: {output_dir}")

        # Paramètres de recherche depuis les variables d'environnement
        location_adzuna = environ.get("LOCATION_ADZUNA", "Paris")
        location_france_travail = environ.get(
            "LOCATION_FRANCE_TRAVAIL", "75"
        )  # Code département Paris

        max_results_per_source = int(environ.get("MAX_RESULTS_PER_SOURCE", "10000"))
        max_days_old = int(environ.get("MAX_DAYS_OLD", "1"))

        CODE_ROME = environ.get(
            "CODE_ROME", "M1805"
        )  # Par défaut: Développement informatique

        logger.info("Paramètres de recherche:")
        logger.info(f"- Code ROME: {CODE_ROME}")
        logger.info(f"- Location Adzuna: {location_adzuna}")
        logger.info(f"- Location France Travail: {location_france_travail}")
        logger.info(f"- Max résultats par source: {max_results_per_source}")
        logger.info(f"- Max de jour à récupérer: {max_days_old}")

        # Initialiser le normalisateur avec gestion d'erreurs
        try:
            async with JobDataNormalizer(
                adzuna_app_id, adzuna_app_key, france_travail_id, france_travail_key
            ) as normalizer:
                start_time = datetime.now()
                logger.info(
                    f"Début de la récupération des données: {start_time.strftime('%H:%M:%S')}"
                )

                # Récupérer et normaliser les données des deux sources
                normalized_jobs = await normalizer.fetch_and_normalize_all(
                    code_rome=CODE_ROME,
                    location_adzuna=location_adzuna,
                    location_france_travail=location_france_travail,
                    max_results=max_results_per_source,
                    max_days_old=max_days_old,
                )

                if not normalized_jobs:
                    logger.warning(
                        "Aucune offre d'emploi récupérée. Vérifiez les paramètres."
                    )
                    return

                # Durée de la récupération
                fetch_duration = datetime.now() - start_time
                logger.info(
                    f"Récupération terminée en {fetch_duration.total_seconds():.1f} secondes"
                )
                logger.info(f"Nombre d'offres récupérées: {len(normalized_jobs)}")

                # Créer un DataFrame pandas
                start_df_time = datetime.now()
                df = normalizer.create_dataframe(normalized_jobs)
                df_duration = datetime.now() - start_df_time
                logger.info(
                    f"Création du DataFrame en {df_duration.total_seconds():.1f} secondes"
                )

                # Indexer pour la performance
                df.reset_index(drop=True, inplace=True)

                # Analyse des données pour les statistiques
                analysis = normalizer.process_and_analyze(df)
                logger.info(
                    f"Analyse complétée: {analysis['total_offers']} offres analysées"
                )

                # Sauvegarder les statistiques
                try:
                    import json

                    stats_path = (
                        output_dir / f"stats_{datetime.now().strftime('%Y%m%d')}.json"
                    )
                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(analysis, f, indent=2, default=str)
                    logger.info(f"Statistiques sauvegardées dans {stats_path}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde des statistiques: {e}")

                # Sauvegarder toutes les offres dans un seul fichier CSV
                start_save_time = datetime.now()
                logger.info("Début de la sauvegarde des données")

                # Définir un nom de fichier avec la date et le nombre d'offres
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                all_jobs_filepath = normalizer.save_to_csv(
                    df,
                    output_dir / f"all_jobs_{date_str}_offers",
                    compress=True,  # Utiliser la compression pour économiser de l'espace
                )

                if all_jobs_filepath:
                    logger.info(f"Fichier principal sauvegardé: {all_jobs_filepath}")
                else:
                    logger.error("Échec de la sauvegarde du fichier principal")

                # Sauvegarder les offres de chaque source séparément pour plus de flexibilité
                for source in ["adzuna", "france_travail"]:
                    source_df = df[df["source"] == source]
                    if len(source_df) > 0:
                        logger.info(f"Sauvegarde de {len(source_df)} offres {source}")
                        source_filepath = normalizer.save_to_csv(
                            source_df,
                            output_dir
                            / f"{source}_jobs_{date_str}_{len(source_df)}_offers",
                        )
                        if source_filepath:
                            logger.info(
                                f"Fichier {source} sauvegardé: {source_filepath}"
                            )
                        else:
                            logger.error(f"Échec de la sauvegarde du fichier {source}")

                # Durée totale de l'opération
                save_duration = datetime.now() - start_save_time
                total_duration = datetime.now() - start_time
                logger.info(
                    f"Sauvegarde terminée en {save_duration.total_seconds():.1f} secondes"
                )
                logger.info(
                    f"Opération totale terminée en {total_duration.total_seconds():.1f} secondes"
                )

                # Afficher un résumé final
                print("\n" + "=" * 50)
                print(f"RÉSUMÉ DE L'OPÉRATION")
                print("=" * 50)
                print(f"Total des offres traitées: {len(df)}")
                print(f"  - Adzuna: {len(df[df['source'] == 'adzuna'])}")
                print(
                    f"  - France Travail: {len(df[df['source'] == 'france_travail'])}"
                )
                print(f"\nFichier principal: {all_jobs_filepath}")
                print(f"Statistiques: {stats_path}")
                print("=" * 50 + "\n")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'initialisation ou de l'utilisation du normalisateur: {e}"
            )
            import traceback

            logger.error(traceback.format_exc())
            raise

    except Exception as e:
        logger.critical(f"Erreur critique dans la fonction principale: {e}")
        import traceback

        logger.critical(traceback.format_exc())
        sys.exit(1)


def init_argparse():
    """Initialise et retourne le parseur d'arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalisation des données d'emploi Adzuna et France Travail"
    )

    parser.add_argument(
        "--code-rome", type=str, help="Code(s) ROME séparés par des virgules"
    )
    parser.add_argument(
        "--location-adzuna", type=str, help="Localisation pour Adzuna (ville, région)"
    )
    parser.add_argument(
        "--location-france-travail",
        type=str,
        help="Code INSEE ou département pour France Travail",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Nombre maximum de résultats par source",
    )
    parser.add_argument(
        "--max-days-old",
        type=int,
        default=1,
        help="Nombre maximum de jour récupéré",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Répertoire de sortie pour les fichiers CSV"
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compresser les fichiers CSV en gzip"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Ne génère que les statistiques sans sauvegarder les CSV",
    )

    return parser


if __name__ == "__main__":
    # Analyser les arguments de ligne de commande
    parser = init_argparse()
    args = parser.parse_args()

    # Définir les variables d'environnement à partir des arguments CLI
    if args.code_rome:
        environ["CODE_ROME"] = args.code_rome
    if args.location_adzuna:
        environ["LOCATION_ADZUNA"] = args.location_adzuna
    if args.location_france_travail:
        environ["LOCATION_FRANCE_TRAVAIL"] = args.location_france_travail
    if args.max_results:
        environ["MAX_RESULTS_PER_SOURCE"] = args.max_results
    if args.max_results:
        environ["MAX_DAYS_OLD"] = args.max_days_old
    if args.output_dir:
        environ["OUTPUT_DIR"] = args.output_dir

    # Exécuter la fonction principale
    asyncio.run(main())
