#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de matching entre offres d'emploi et candidats utilisant SBERT, TF-IDF et Cross-Encoder
"""

import os
import pandas as pd
import numpy as np
import gzip
import re
import string
import mlflow
import mlflow.pyfunc
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
import argparse
from datetime import datetime

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Cross-Encoder non disponible. Installation recommandée: pip install sentence-transformers")

try:
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    try:
        french_stopwords = stopwords.words('french')
        english_stopwords = stopwords.words('english')
    except:
        import nltk
        nltk.download('stopwords')
        french_stopwords = stopwords.words('french')
        english_stopwords = stopwords.words('english')
    ALL_STOPWORDS = set(french_stopwords + english_stopwords)
except ImportError:
    NLTK_AVAILABLE = False
    ALL_STOPWORDS = set()
    print("NLTK non disponible. Installation recommandée: pip install nltk")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobMatchingModel(mlflow.pyfunc.PythonModel):
    """Modèle de matching offres-candidats avec SBERT et TF-IDF"""
    
    def __init__(self, model_name="distiluse-base-multilingual-cased-v1", use_tfidf=True, tfidf_weight=0.3):
        self.model_name = model_name
        self.model = None
        self.use_tfidf = use_tfidf
        self.tfidf_weight = tfidf_weight
        self.tfidf_vectorizer = None
        self.scaler = MinMaxScaler()
        
        # Poids des différents facteurs dans le score
        self.weights = {
            "text_similarity": 0.6,
            "location_match": 0.2,
            "contract_match": 0.1,
            "salary_match": 0.1
        }
        
        # Détection du GPU
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logger.info(f"CUDA disponible. GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA non disponible. Utilisation du CPU.")
    
    def load_context(self, context):
        """Charge le modèle lors de la restauration"""
        self.model = SentenceTransformer(self.model_name)
        
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85
            )
    
    def predict(self, context, model_input):
        """Prédit les matchs entre offres et candidats"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            
        if self.use_tfidf and self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.85
            )
        
        job_offers = model_input["job_offers"]
        candidates = model_input["candidates"]
        
        return self._calculate_matches(job_offers, candidates)
    
    def _preprocess_text(self, text):
        """Prétraite le texte pour TF-IDF"""
        if not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression de la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Suppression des chiffres
        text = re.sub(r'\d+', '', text)
        
        # Suppression des stopwords
        if ALL_STOPWORDS:
            tokens = text.split()
            text = ' '.join([t for t in tokens if t not in ALL_STOPWORDS])
            
        return text
    
    def _calculate_matches(self, job_offers, candidates):
        """Calcule les matchs entre offres et candidats"""
        # Vérifier les colonnes requises
        required_columns = {
            'jobs': ['id_offre', 'description'],
            'candidates': ['id_candidat']
        }
        
        for col in required_columns['jobs']:
            if col not in job_offers.columns:
                raise ValueError(f"Colonne manquante dans les offres: {col}")
                
        for col in required_columns['candidates']:
            if col not in candidates.columns:
                raise ValueError(f"Colonne manquante dans les candidats: {col}")
        
        # Préparer les données
        job_descriptions = []
        job_ids = []
        candidate_profiles = []
        candidate_ids = []
        
        # Extraire les textes des offres
        for _, job in job_offers.iterrows():
            job_text = f"{job.get('titre', '')} {job.get('description', '')} {job.get('categorie', '')} {job.get('secteur', '')}"
            job_descriptions.append(job_text)
            job_ids.append(job['id_offre'])
        
        # Extraire les textes des candidats
        for _, candidate in candidates.iterrows():
            if 'profil_texte' in candidate and candidate['profil_texte']:
                candidate_text = candidate['profil_texte']
            else:
                candidate_text = f"{candidate.get('competences', '')} {candidate.get('domaines', '')} {candidate.get('metiers', '')}"
            
            candidate_profiles.append(candidate_text)
            candidate_ids.append(candidate['id_candidat'])
        
        # Matrice de similarité TF-IDF
        tfidf_similarity = None
        if self.use_tfidf:
            logger.info("Calcul des vecteurs TF-IDF")
            try:
                # Prétraitement des textes
                preprocessed_jobs = [self._preprocess_text(text) for text in job_descriptions]
                preprocessed_candidates = [self._preprocess_text(text) for text in candidate_profiles]
                
                # Combinaison pour l'apprentissage
                all_texts = preprocessed_jobs + preprocessed_candidates
                
                # Fit et transform
                self.tfidf_vectorizer.fit(all_texts)
                job_vectors = self.tfidf_vectorizer.transform(preprocessed_jobs)
                candidate_vectors = self.tfidf_vectorizer.transform(preprocessed_candidates)
                
                # Calcul similarité
                tfidf_similarity = cosine_similarity(candidate_vectors, job_vectors)
            except Exception as e:
                logger.error(f"Erreur lors du calcul TF-IDF: {str(e)}")
                self.use_tfidf = False
        
        # Encodage SBERT
        logger.info(f"Encodage de {len(job_descriptions)} offres et {len(candidate_profiles)} candidats")
        job_embeddings = self.model.encode(job_descriptions, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidate_profiles, convert_to_tensor=True)
        
        # Calcul similarité SBERT
        sbert_similarity = util.pytorch_cos_sim(candidate_embeddings, job_embeddings)
        if sbert_similarity.is_cuda:
            sbert_similarity = sbert_similarity.cpu()
        sbert_similarity = sbert_similarity.numpy()
        
        # Combinaison des similarités
        if self.use_tfidf and tfidf_similarity is not None:
            logger.info("Combinaison des scores SBERT et TF-IDF")
            text_similarity = (1 - self.tfidf_weight) * sbert_similarity + self.tfidf_weight * tfidf_similarity
        else:
            text_similarity = sbert_similarity
        
        # Calcul des matchs
        all_matches = []
        for i, candidate_id in enumerate(candidate_ids):
            candidate = candidates[candidates['id_candidat'] == candidate_id].iloc[0]
            
            for j, job_id in enumerate(job_ids):
                job = job_offers[job_offers['id_offre'] == job_id].iloc[0]
                
                # Score de similarité textuelle
                text_sim = text_similarity[i, j]
                
                # Autres scores
                location_match = self._calc_location_match(candidate, job)
                contract_match = self._calc_contract_match(candidate, job)
                salary_match = self._calc_salary_match(candidate, job)
                
                # Score global
                overall_score = (
                    self.weights["text_similarity"] * text_sim +
                    self.weights["location_match"] * location_match +
                    self.weights["contract_match"] * contract_match +
                    self.weights["salary_match"] * salary_match
                )
                
                # Ajouter à la liste
                all_matches.append({
                    "id_candidat": candidate_id,
                    "id_offre": job_id,
                    "score_global": overall_score,
                    "score_text": text_sim,
                    "score_location": location_match,
                    "score_contract": contract_match,
                    "score_salary": salary_match,
                    "score_sbert": float(sbert_similarity[i, j]),
                    "score_tfidf": float(tfidf_similarity[i, j] if self.use_tfidf and tfidf_similarity is not None else 0.0)
                })
        
        # Créer DataFrame et normaliser les scores
        matches_df = pd.DataFrame(all_matches)
        if not matches_df.empty:
            matches_df["score_global"] = self.scaler.fit_transform(matches_df[["score_global"]])
        
        return matches_df
    
    def _calc_location_match(self, candidate, job):
        """Calcule la correspondance de localisation"""
        if candidate.get('id_lieu') == job.get('id_lieu'):
            return 1.0
        return 0.0
    
    def _calc_contract_match(self, candidate, job):
        """Calcule la correspondance de contrat"""
        if candidate.get('id_contrat') == job.get('id_contrat'):
            return 1.0
        return 0.0
    
    def _calc_salary_match(self, candidate, job):
        """Calcule la correspondance de salaire"""
        candidate_min = candidate.get('salaire_min')
        job_salary = job.get('salaire')
        
        if pd.isna(candidate_min) or pd.isna(job_salary):
            return 0.5
        
        if job_salary >= candidate_min:
            return 1.0
        
        ratio = job_salary / candidate_min
        return max(0, min(1, ratio))


class JobMatchingSystem:
    """Système complet de matching offres-candidats"""
    
    def __init__(self, offers_file, candidates_file, 
                 model_name="distiluse-base-multilingual-cased-v1",
                 use_tfidf=True, tfidf_weight=0.3,
                 use_cross_encoder=False, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 cross_encoder_weight=0.7):
        self.offers_file = offers_file
        self.candidates_file = candidates_file
        self.model_name = model_name
        
        # Configuration TF-IDF
        self.use_tfidf = use_tfidf
        self.tfidf_weight = tfidf_weight
        
        # Configuration Cross-Encoder
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder_weight = cross_encoder_weight
        self.cross_encoder = None
        
        # Modèle bi-encoder
        self.matching_model = JobMatchingModel(model_name, use_tfidf, tfidf_weight)
        
        # Initialisation des modèles
        if self.matching_model.model is None:
            self.matching_model.model = SentenceTransformer(model_name)
        
        if self.use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Cross-Encoder initialisé: {cross_encoder_model}")
            except Exception as e:
                logger.error(f"Erreur d'initialisation du Cross-Encoder: {str(e)}")
                self.use_cross_encoder = False
    
    def load_data(self):
        """Charge les données des fichiers"""
        # Vérifier les fichiers
        if not os.path.exists(self.offers_file):
            raise FileNotFoundError(f"Fichier non trouvé: {self.offers_file}")
        
        if not os.path.exists(self.candidates_file):
            raise FileNotFoundError(f"Fichier non trouvé: {self.candidates_file}")
        
        # Charger les offres
        logger.info(f"Chargement des offres: {self.offers_file}")
        if self.offers_file.endswith('.gz'):
            with gzip.open(self.offers_file, 'rt') as f:
                job_offers = pd.read_csv(f)
        else:
            job_offers = pd.read_csv(self.offers_file)
        
        # Charger les candidats
        logger.info(f"Chargement des candidats: {self.candidates_file}")
        candidates = pd.read_csv(self.candidates_file)
        
        # Prétraitement
        job_offers = self._preprocess_jobs(job_offers)
        candidates = self._preprocess_candidates(candidates)
        
        logger.info(f"Chargé {len(job_offers)} offres et {len(candidates)} candidats")
        
        return job_offers, candidates
    
    def _preprocess_jobs(self, df):
        """Prétraite les offres d'emploi"""
        processed_df = df.copy()

        # Limiter à 2000 lignes maximum
        if len(df) > 2000:
            logger.info(f"Limitation du fichier d'offres à 2000 lignes (sur {len(df)} disponibles)")
            df = df.head(2000)
        
        processed_df = df.copy()
        
        # Afficher les colonnes pour débug
        logger.info(f"Colonnes des offres: {processed_df.columns.tolist()}")
        
        # Mapping des colonnes
        columns_mapping = {
            'id': 'id_offre',
            'id_local': 'id_offre',
            'ID_LOCAL': 'id_offre',
            'title': 'titre',
            'TITLE': 'titre',
            'company_name': 'entreprise_nom',
            'location_name': 'ville',
            'contract_type': 'type_contrat',
            'TYPE_CONTRAT': 'type_contrat',
            'working_hours': 'temps_travail',
            'salary_min': 'salaire',
            'category': 'categorie',
            'sector': 'secteur',
            'skills': 'competences',
            'COMPETENCES': 'competences',
            'remote_work': 'teletravail',
            'DESCRIPTION': 'description',
            'experience_required': 'experience'
        }
        
        # Appliquer le mapping
        rename_dict = {old: new for old, new in columns_mapping.items() if old in processed_df.columns}
        processed_df = processed_df.rename(columns=rename_dict)
        
        # Vérifier l'identifiant
        if 'id_offre' not in processed_df.columns:
            if 'id' in processed_df.columns:
                processed_df['id_offre'] = processed_df['id']
            elif 'id_local' in df.columns:
                processed_df['id_offre'] = df['id_local']
            elif 'ID_LOCAL' in df.columns:
                processed_df['id_offre'] = df['ID_LOCAL']
            else:
                processed_df['id_offre'] = range(1, len(processed_df) + 1)
                logger.warning("Aucun identifiant trouvé, création d'identifiants séquentiels")
        
        # Vérifier la description
        if 'description' not in processed_df.columns and 'DESCRIPTION' in processed_df.columns:
            processed_df['description'] = processed_df['DESCRIPTION']
        elif 'description' not in processed_df.columns:
            processed_df['description'] = ''
        
        # Vérifier le titre
        if 'titre' not in processed_df.columns and 'TITLE' in processed_df.columns:
            processed_df['titre'] = processed_df['TITLE']
        elif 'titre' not in processed_df.columns:
            processed_df['titre'] = ''
        
        # Remplir les valeurs manquantes
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].fillna('')
        
        return processed_df
    
    def _preprocess_candidates(self, df):
        """Prétraite les candidats"""
        processed_df = df.copy()
        
        # Afficher les colonnes pour débug
        logger.info(f"Colonnes des candidats: {processed_df.columns.tolist()}")
        
        # Vérifier l'identifiant
        if 'id_candidat' not in processed_df.columns:
            if 'id' in processed_df.columns:
                processed_df['id_candidat'] = processed_df['id']
            else:
                processed_df['id_candidat'] = range(1, len(processed_df) + 1)
                logger.warning("Aucun identifiant trouvé, création d'identifiants séquentiels")
        
        # Créer les colonnes de compétences si absentes
        for col_name in ['competences', 'domaines', 'metiers']:
            if col_name not in processed_df.columns:
                source_col = f'id_{col_name[:-1]}'  # ex: 'id_competence' pour 'competences'
                if source_col in processed_df.columns:
                    processed_df[col_name] = processed_df[source_col].fillna('')
                else:
                    processed_df[col_name] = ''
        
        # Créer un profil texte
        available_columns = [col for col in ['nom', 'prenom', 'competences', 'domaines', 'metiers'] 
                            if col in processed_df.columns]
        
        if available_columns:
            profil_texts = []
            for _, row in processed_df.iterrows():
                parts = []
                for col in available_columns:
                    if pd.notna(row[col]) and row[col] != '':
                        parts.append(str(row[col]))
                profil_texts.append(' '.join(parts))
            
            processed_df['profil_texte'] = profil_texts
        else:
            processed_df['profil_texte'] = 'candidat'
            logger.warning("Aucune colonne disponible pour créer un profil texte")
        
        # Remplir les valeurs manquantes
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].fillna('')
        
        return processed_df
    
    def run_matching(self, job_offers, candidates, top_n=10):
        """Exécute le processus de matching"""
        logger.info("Début du processus de matching")
        
        # Phase 1: Matching initial avec Bi-Encoder
        model_input = {
            "job_offers": job_offers,
            "candidates": candidates
        }
        
        logger.info("Phase 1: Matching avec Bi-Encoder (SBERT/TF-IDF)")
        biencoder_matches = self.matching_model.predict(None, model_input)
        
        # Si pas de Cross-Encoder, retourner les résultats du Bi-Encoder
        if not self.use_cross_encoder or self.cross_encoder is None:
            logger.info("Utilisation des résultats du Bi-Encoder uniquement")
            return self._organize_biencoder_results(biencoder_matches, job_offers, candidates, top_n)
        
        # Phase 2: Affinement avec Cross-Encoder
        logger.info("Phase 2: Affinement avec Cross-Encoder")
        
        # Nombre de candidats à réévaluer
        n_candidates_for_cross = min(500, len(job_offers))
        
        # Résultats finaux
        final_results = {}
        
        # Pour chaque candidat
        for candidate_id in candidates['id_candidat'].unique():
            candidate = candidates[candidates['id_candidat'] == candidate_id].iloc[0]
            
            # Texte du candidat
            candidate_text = candidate['profil_texte']
            
            # Meilleurs matchs du Bi-Encoder
            candidate_matches = biencoder_matches[biencoder_matches['id_candidat'] == candidate_id]
            top_matches = candidate_matches.sort_values('score_global', ascending=False).head(n_candidates_for_cross)
            
            # Préparer paires pour Cross-Encoder
            pairs = []
            job_ids = []
            bi_scores = []
            
            for _, match in top_matches.iterrows():
                job_id = match['id_offre']
                bi_score = match['score_global']
                
                try:
                    job = job_offers[job_offers['id_offre'] == job_id].iloc[0]
                    job_text = f"{job.get('titre', '')} {job.get('description', '')}"
                    
                    pairs.append([candidate_text, job_text])
                    job_ids.append(job_id)
                    bi_scores.append(bi_score)
                except:
                    continue
            
            # Pas de matchs? Passer au candidat suivant
            if not pairs:
                final_results[candidate_id] = []
                continue
            
            # Évaluation Cross-Encoder
            logger.info(f"Évaluation de {len(pairs)} paires pour candidat {candidate_id}")
            try:
                cross_scores = self.cross_encoder.predict(pairs)
                
                # Normalisation
                if not isinstance(cross_scores, np.ndarray):
                    cross_scores = np.array(cross_scores)
                
                if cross_scores.max() > 1 or cross_scores.min() < 0:
                    cross_scores = (cross_scores - cross_scores.min()) / (cross_scores.max() - cross_scores.min() + 1e-10)
            except Exception as e:
                logger.error(f"Erreur Cross-Encoder: {str(e)}")
                cross_scores = np.ones(len(pairs))
            
            # Combinaison des scores
            combined_scores = []
            for i, (job_id, bi_score, cross_score) in enumerate(zip(job_ids, bi_scores, cross_scores)):
                # Score final pondéré
                bi_weight = 1.0 - self.cross_encoder_weight
                final_score = bi_weight * bi_score + self.cross_encoder_weight * float(cross_score)
                
                job = job_offers[job_offers['id_offre'] == job_id].iloc[0]
                
                combined_scores.append({
                    'id_offre': job_id,
                    'score_bi': float(bi_score),
                    'score_cross': float(cross_score),
                    'score_global': float(final_score),
                    'job_info': {
                        "description": job.get('description', ''),
                        "titre": job.get('titre', ''),
                        "entreprise": job.get('entreprise_nom', ''),
                        "lieu": f"{job.get('ville', '')}, {job.get('departement', '')}",
                        "contrat": job.get('type_contrat', ''),
                        "salaire": job.get('salaire', None)
                    }
                })
            
            # Trier et prendre les top_n
            sorted_scores = sorted(combined_scores, key=lambda x: x['score_global'], reverse=True)[:top_n]
            final_results[candidate_id] = sorted_scores
        
        logger.info(f"Matching terminé pour {len(candidates)} candidats")
        return final_results
    
    def _organize_biencoder_results(self, matches_df, job_offers, candidates, top_n):
        """Organise les résultats du Bi-Encoder"""
        results = {}
        
        for candidate_id in candidates['id_candidat'].unique():
            candidate_matches = matches_df[matches_df['id_candidat'] == candidate_id]
            top_matches = candidate_matches.sort_values('score_global', ascending=False).head(top_n)
            
            candidate_results = []
            for _, match in top_matches.iterrows():
                job_id = match['id_offre']
                try:
                    job = job_offers[job_offers['id_offre'] == job_id].iloc[0]
                    
                    candidate_results.append({
                        "id_offre": job_id,
                        "score_bi": float(match['score_global']),
                        "score_cross": 0.0,
                        "score_global": float(match['score_global']),
                        "job_info": {
                            "description": job.get('description', ''),
                            "titre": job.get('titre', ''),
                            "entreprise": job.get('entreprise_nom', ''),
                            "lieu": f"{job.get('ville', '')}, {job.get('departement', '')}",
                            "contrat": job.get('type_contrat', ''),
                            "salaire": job.get('salaire', None)
                        }
                    })
                except:
                    continue
            
            results[candidate_id] = candidate_results
        
        return results
    
    def save_results(self, results, output_path):
        """Sauvegarde les résultats dans un fichier CSV"""
        rows = []
        
        for candidate_id, matches in results.items():
            for match in matches:
                row = {
                    "id_candidat": candidate_id,
                    "id_offre": match["id_offre"],
                    "score_global": match["score_global"],
                    "score_bi_encoder": match.get("score_bi", match["score_global"]),
                    "score_cross_encoder": match.get("score_cross", 0.0),
                    "score_sbert": match.get("score_sbert", 0.0),
                    "score_tfidf": match.get("score_tfidf", 0.0),
                    "titre": match["job_info"].get("titre", ""),
                    "entreprise": match["job_info"].get("entreprise", ""),
                    "lieu": match["job_info"].get("lieu", ""),
                    "contrat": match["job_info"].get("contrat", ""),
                    "salaire": match["job_info"].get("salaire", "")
                }
                rows.append(row)
        
        # Créer et sauvegarder DataFrame
        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Résultats sauvegardés: {output_path}")
    
    def run_pipeline(self, output_dir="./output", top_n=10):
        """Exécute le pipeline complet"""
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp pour les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configurer MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("job_matching_sbert")
        
        # Démarrer MLflow run
        with mlflow.start_run(run_name=f"matching_run_{timestamp}"):
            # Charger données
            job_offers, candidates = self.load_data()
            
            # Enregistrer paramètres
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("use_tfidf", self.use_tfidf)
            if self.use_tfidf:
                mlflow.log_param("tfidf_weight", self.tfidf_weight)
            mlflow.log_param("use_cross_encoder", self.use_cross_encoder)
            if self.use_cross_encoder:
                mlflow.log_param("cross_encoder_model", self.cross_encoder_model)
                mlflow.log_param("cross_encoder_weight", self.cross_encoder_weight)
            mlflow.log_param("top_n", top_n)
            mlflow.log_param("nb_job_offers", len(job_offers))
            mlflow.log_param("nb_candidates", len(candidates))
            
            # Exécution matching
            results = self.run_matching(job_offers, candidates, top_n)
            
            # Calculer métriques
            metrics = {
                "global": [],
                "bi_encoder": [],
                "cross_encoder": []
            }
            
            for _, matches in results.items():
                if matches:
                    metrics["global"].append(sum(match["score_global"] for match in matches) / len(matches))
                    metrics["bi_encoder"].append(sum(match.get("score_bi", match["score_global"]) for match in matches) / len(matches))
                    metrics["cross_encoder"].append(sum(match.get("score_cross", 0) for match in matches) / len(matches))
            
            # Enregistrer métriques
            if metrics["global"]:
                mlflow.log_metric("mean_match_score", sum(metrics["global"]) / len(metrics["global"]))
                mlflow.log_metric("max_match_score", max(metrics["global"]))
                mlflow.log_metric("min_match_score", min(metrics["global"]))
                
                if self.use_cross_encoder:
                    mlflow.log_metric("mean_bi_encoder_score", sum(metrics["bi_encoder"]) / len(metrics["bi_encoder"]))
                    mlflow.log_metric("mean_cross_encoder_score", sum(metrics["cross_encoder"]) / len(metrics["cross_encoder"]))
            
            # Sauvegarder résultats
            output_path = os.path.join(output_dir, f"matching_results_{timestamp}.csv")
            self.save_results(results, output_path)
            
            # Enregistrer modèle
            mlflow.pyfunc.log_model(
                artifact_path="models",
                python_model=self.matching_model,
                conda_env={
                    "channels": ["defaults", "pytorch", "huggingface"],
                    "dependencies": [
                        "python=3.9",
                        "pip",
                        {"pip": [
                            "sentence-transformers>=2.0.0",
                            "mlflow>=2.0.0",
                            "pandas>=1.3.0",
                            "numpy>=1.20.0",
                            "scikit-learn>=1.0.0",
                            "torch>=1.9.0",
                            "nltk>=3.6.0"
                        ]}
                    ],
                    "name": "job_matching_env"
                }
            )
            
            # Sauvegarder données
            data_path = os.path.join(output_dir, f"job_offers_{timestamp}.csv")
            job_offers.to_csv(data_path, index=False)
            mlflow.log_artifact(data_path)
            
            data_path = os.path.join(output_dir, f"candidates_{timestamp}.csv")
            candidates.to_csv(data_path, index=False)
            mlflow.log_artifact(data_path)
            
            logger.info(f"Pipeline terminé. Résultats dans MLflow et {output_dir}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Matching offres-candidats avec SBERT/TF-IDF/Cross-Encoder")
    
    # Chemins des fichiers
    parser.add_argument("--offers", type=str, 
                       default="/home/arnaudrk/arnaudUBUNTU/mar25_bootcamp_de_job_market/MLFlow/data/OnBigTable/one_big_table.csv.gz",
                       help="Chemin du fichier d'offres d'emploi")
    parser.add_argument("--candidates", type=str,
                       default="/home/arnaudrk/arnaudUBUNTU/mar25_bootcamp_de_job_market/pipeline/src/data/RAW_CANDIDAT.csv",
                       help="Chemin du fichier de candidats")
    
    # Configuration générale
    parser.add_argument("--model", type=str, default="distiluse-base-multilingual-cased-v1",
                       help="Modèle SBERT à utiliser")
    parser.add_argument("--output", type=str, default="./output",
                       help="Répertoire de sortie")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Nombre de matchs par candidat")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:8080",
                       help="URI du serveur MLflow")
    
    # Configuration TF-IDF
    parser.add_argument("--use-tfidf", action="store_true",
                       help="Utiliser TF-IDF")
    parser.add_argument("--tfidf-weight", type=float, default=0.3,
                       help="Poids de TF-IDF (0-1)")
    
    # Configuration Cross-Encoder
    parser.add_argument("--use-cross-encoder", action="store_true",
                       help="Utiliser Cross-Encoder")
    parser.add_argument("--cross-encoder-model", type=str, 
                       default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Modèle Cross-Encoder")
    parser.add_argument("--cross-encoder-weight", type=float, default=0.7,
                       help="Poids du Cross-Encoder (0-1)")
    
    args = parser.parse_args()
    
    # Configurer MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    try:
        # Initialiser système
        matching_system = JobMatchingSystem(
            offers_file=args.offers,
            candidates_file=args.candidates,
            model_name=args.model,
            use_tfidf=args.use_tfidf,
            tfidf_weight=args.tfidf_weight,
            use_cross_encoder=args.use_cross_encoder,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_weight=args.cross_encoder_weight
        )
        
        # Exécuter pipeline
        matching_system.run_pipeline(
            output_dir=args.output,
            top_n=args.top_n
        )
        
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())