# Système de Matching Offres d'Emploi - Candidats

Ce système utilise l'apprentissage profond et le traitement du langage naturel pour faire correspondre des offres d'emploi avec des profils de candidats. Il implémente une approche hybride combinant SBERT (Sentence-BERT), TF-IDF et Cross-Encoder pour une précision optimale.

## Table des matières

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Fonctionnement](#fonctionnement)
- [Options et paramètres](#options-et-paramètres)
- [Modèles disponibles](#modèles-disponibles)
- [Métriques et caractéristiques](#métriques-et-caractéristiques)
- [Résultats attendus](#résultats-attendus)
- [Interprétation des résultats](#interprétation-des-résultats)
- [Exemples d'utilisation](#exemples-dutilisation)

## Prérequis

- Python 3.8+
- CUDA compatible GPU (recommandé mais non obligatoire)
- Accès à un serveur MLflow (par défaut: http://127.0.0.1:8080)

## Installation

1. Clonez ce dépôt:
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]
```

2. Installez les dépendances:
```bash
pip install -r requirements.txt
```

3. Téléchargez les ressources NLTK (stopwords):
```bash
python -c "import nltk; nltk.download('stopwords')"
```

4. (Optionnel) Démarrez le serveur MLflow:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Fonctionnement

Le script implémente un pipeline de matching en trois phases:

### 1. Phase d'encodage Bi-Encoder (SBERT)
- Transforme les descriptions d'offres et les profils de candidats en vecteurs numériques
- Capture la sémantique des textes indépendamment des mots exacts utilisés

### 2. Phase d'analyse TF-IDF (optionnelle)
- Complète SBERT en identifiant les mots-clés spécifiques et importants
- Améliore la détection des termes techniques ou compétences précises

### 3. Phase d'affinement Cross-Encoder (optionnelle)
- Analyse approfondie des meilleures paires offre-candidat
- Évalue la pertinence contextuelle des correspondances

### Architecture du système:
```
Données d'entrée
    ↓
Prétraitement
    ↓
┌─────────────────────┐
│  Phase 1: Bi-Encoder │
│   ┌───────────────┐ │
│   │ SBERT Encodage │ │        ┌───────────────────┐
│   └───────┬───────┘ │        │  Phase 2: TF-IDF   │
│           │         │◄───────┤     (optionnel)    │
│   ┌───────▼───────┐ │        └───────────────────┘
│   │ Score initial  │ │
│   └───────────────┘ │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Filtrage des     │
│ meilleures paires   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Phase 3: Cross-Enc. │
│     (optionnel)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Résultats finaux   │
└─────────────────────┘
```

## Options et paramètres

### Paramètres des fichiers
- `--offers` : Chemin vers le fichier des offres d'emploi (CSV ou CSV.GZ)
- `--candidates` : Chemin vers le fichier des candidats (CSV)
- `--output` : Répertoire de sortie pour les résultats

### Configuration générale
- `--model` : Modèle SBERT à utiliser (défaut: "distiluse-base-multilingual-cased-v1")
- `--top-n` : Nombre de correspondances à retourner par candidat (défaut: 10)
- `--mlflow-uri` : URI du serveur MLflow (défaut: "http://127.0.0.1:8080")

### Configuration TF-IDF
- `--use-tfidf` : Active l'utilisation de TF-IDF pour enrichir l'analyse
- `--tfidf-weight` : Poids de TF-IDF dans le score textuel (défaut: 0.3)

### Configuration Cross-Encoder
- `--use-cross-encoder` : Active l'utilisation du Cross-Encoder pour affiner les résultats
- `--cross-encoder-model` : Modèle Cross-Encoder à utiliser (défaut: "cross-encoder/ms-marco-MiniLM-L-6-v2")
- `--cross-encoder-weight` : Poids du Cross-Encoder dans le score final (défaut: 0.7)

## Modèles disponibles

### Modèles SBERT (Bi-Encoder)
- **distiluse-base-multilingual-cased-v1** (défaut) : Bon équilibre entre performance et vitesse, multilingual
- **paraphrase-multilingual-MiniLM-L12-v2** : Plus précis, multilingual, mais plus lent
- **paraphrase-distilroberta-base-v2** : Optimisé pour l'anglais, très performant si les données sont en anglais

### Modèles Cross-Encoder
- **cross-encoder/ms-marco-MiniLM-L-6-v2** (défaut) : Bon équilibre performance/vitesse
- **cross-encoder/ms-marco-TinyBERT-L-2-v2** : Plus rapide, moins précis
- **cross-encoder/ms-marco-roberta-base** : Plus précis, plus lent

## Métriques et caractéristiques

### Métriques suivies dans MLflow
- `mean_match_score` : Score moyen de matching pour tous les candidats
- `max_match_score` : Score maximum obtenu
- `min_match_score` : Score minimum obtenu
- `mean_bi_encoder_score` : Score moyen du Bi-Encoder (si Cross-Encoder est activé)
- `mean_cross_encoder_score` : Score moyen du Cross-Encoder (si activé)
- `mean_sbert_score` : Score moyen du modèle SBERT (si TF-IDF est activé)
- `mean_tfidf_score` : Score moyen de TF-IDF (si activé)

### Facteurs de matching
Le script prend en compte plusieurs critères pour calculer le score global:
- Similarité textuelle (60%): Mesure à quel point les descriptions et profils correspondent
- Correspondance de localisation (20%): Si le candidat et l'offre partagent la même région
- Correspondance de contrat (10%): Si le type de contrat correspond aux préférences
- Correspondance de salaire (10%): Si le salaire proposé répond aux attentes minimales

## Résultats attendus

Le script produit:

1. **Fichier CSV** contenant pour chaque candidat les N meilleures offres correspondantes avec:
   - Identifiants candidat/offre
   - Scores de matching (global, SBERT, TF-IDF, Cross-Encoder selon configuration)
   - Informations sur l'offre (titre, entreprise, lieu, contrat, salaire)

2. **Expérience MLflow** avec:
   - Paramètres utilisés
   - Métriques calculées
   - Modèle sauvegardé et artefacts

## Interprétation des résultats

### Scores de matching
- **Score global > 0.8** : Correspondance excellente
- **Score global 0.6-0.8** : Bonne correspondance
- **Score global 0.4-0.6** : Correspondance moyenne
- **Score global < 0.4** : Correspondance faible

### Analyse de la contribution des différents scores
- **score_sbert** élevé : Bonne correspondance sémantique globale
- **score_tfidf** élevé : Bonne correspondance de mots-clés spécifiques
- **score_cross_encoder** élevé : Bonne pertinence contextuelle

### Exemple d'interprétation

```
id_candidat,id_offre,score_global,score_bi_encoder,score_cross_encoder,score_sbert,score_tfidf,titre,...
0,6081594,0.7171,0.7171,0.0,0.7212,0.0,Data Scientist Confirmé (H/F),...
```

Cette ligne indique:
- Le candidat 0 et l'offre 6081594 ont un score de matching de 0.7171 (bonne correspondance)
- Le matching a été effectué avec le Bi-Encoder uniquement (pas de Cross-Encoder)
- Le score est entièrement basé sur SBERT (pas de TF-IDF)
- Le poste correspondant est "Data Scientist Confirmé (H/F)"

## Exemples d'utilisation

### Utilisation de base (SBERT uniquement)
```bash
python sbert_crossencoder.py
```

### Avec TF-IDF (pour améliorer la détection des compétences spécifiques)
```bash
python sbert_crossencoder.py --use-tfidf
```

### Avec Cross-Encoder (pour une analyse approfondie)
```bash
python sbert_crossencoder.py --use-cross-encoder
```

### Configuration complète (recommandée)
```bash
python sbert_crossencoder.py --use-tfidf --use-cross-encoder
```

### Avec paramètres personnalisés
```bash
python sbert_crossencoder.py --use-tfidf --tfidf-weight 0.4 --use-cross-encoder --cross-encoder-weight 0.8 --top-n 20
```

### Limitation à 2000 offres d'emploi
Le script limite automatiquement le traitement aux 2000 premières offres d'emploi pour optimiser les performances.## Interprétation des résultats

### Scores de matching
- **Score global > 0.8** : Correspondance excellente
- **Score global 0.6-0.8** : Bonne correspondance
- **Score global 0.4-0.6** : Correspondance moyenne
- **Score global < 0.4** : Correspondance faible

### Analyse de la contribution des différents scores
- **score_sbert** élevé : Bonne correspondance sémantique globale
- **score_tfidf** élevé : Bonne correspondance de mots-clés spécifiques
- **score_cross_encoder** élevé : Bonne pertinence contextuelle

### Exemple d'interprétation

```
id_candidat,id_offre,score_global,score_bi_encoder,score_cross_encoder,score_sbert,score_tfidf,titre,...
0,6081594,0.7171,0.7171,0.0,0.7212,0.0,Data Scientist Confirmé (H/F),...
```

Cette ligne indique:
- Le candidat 0 et l'offre 6081594 ont un score de matching de 0.7171 (bonne correspondance)
- Le matching a été effectué avec le Bi-Encoder uniquement (pas de Cross-Encoder)
- Le score est entièrement basé sur SBERT (pas de TF-IDF)
- Le poste correspondant est "Data Scientist Confirmé (H/F)"

## Comprendre les différents scores

Le système génère plusieurs types de scores qui peuvent être interprétés différemment:

### Score SBERT vs Score Bi-Encoder

**Score SBERT**
- Représente le score brut obtenu directement du modèle SBERT (Sentence-BERT) sans aucune modification.
- C'est la similarité cosinus entre les embeddings vectoriels des textes, tels que générés par le modèle SBERT.
- Reflète la similarité sémantique pure entre le profil du candidat et la description de l'offre.

**Score Bi-Encoder**
- Dans le script, le terme "Bi-Encoder" désigne l'**ensemble de la première phase** du pipeline de matching, qui peut inclure:
  - Le score SBERT (toujours présent)
  - Le score TF-IDF (si l'option `--use-tfidf` est activée)
  - Leur combinaison pondérée (si les deux sont présents)

### Relation entre ces scores

1. **Scénario sans TF-IDF** (option `--use-tfidf` non activée):
   ```
   Score Bi-Encoder = Score SBERT
   ```
   Dans ce cas, les deux scores sont identiques car SBERT est la seule méthode d'encodage utilisée.

2. **Scénario avec TF-IDF** (option `--use-tfidf` activée):
   ```
   Score Bi-Encoder = (1 - tfidf_weight) * Score SBERT + tfidf_weight * Score TF-IDF
   ```
   Le score Bi-Encoder devient une combinaison pondérée où:
   - `tfidf_weight` est le poids donné à TF-IDF (par défaut: 0.3)
   - La différence entre Score SBERT et Score Bi-Encoder reflète l'influence de TF-IDF

### Dans le fichier CSV de résultats

- `score_sbert`: Le score pur du modèle SBERT
- `score_tfidf`: Le score pur de l'analyse TF-IDF
- `score_bi_encoder`: Le score combiné de la première phase (SBERT + TF-IDF si activé)
- `score_cross_encoder`: Le score de la seconde phase (si Cross-Encoder est activé)
- `score_global`: Le score final, qui peut être:
  - Égal au `score_bi_encoder` (si Cross-Encoder n'est pas activé)
  - Une combinaison pondérée de `score_bi_encoder` et `score_cross_encoder` (si Cross-Encoder est activé)

### Exemple concret

Considérons ces valeurs hypothétiques:
- `score_sbert = 0.75`
- `score_tfidf = 0.60`
- `tfidf_weight = 0.3`

Le `score_bi_encoder` serait calculé comme:
```
score_bi_encoder = (1 - 0.3) * 0.75 + 0.3 * 0.60
                  = 0.7 * 0.75 + 0.3 * 0.60
                  = 0.525 + 0.18
                  = 0.705
```

Dans ce cas:
- `score_sbert` (0.75) montre une bonne similarité sémantique
- `score_tfidf` (0.60) indique une correspondance moyenne des mots-clés
- `score_bi_encoder` (0.705) combine ces deux aspects, légèrement inférieur au score SBERT pur en raison de l'influence modérée de TF-IDF

### Pourquoi cette distinction est importante

Cette distinction permet d'analyser:
1. La contribution spécifique de chaque technique au score global
2. Si la correspondance est davantage sémantique (score SBERT élevé) ou lexicale (score TF-IDF élevé)
3. L'effet des différentes pondérations sur le score final

En examinant ces différents scores, vous pouvez déterminer si un bon matching est dû à une compréhension sémantique générale ou à des correspondances précises de mots-clés, ce qui est particulièrement utile dans le contexte du matching offres d'emploi-candidats.

## Exemples d'utilisation

### Utilisation de base (SBERT uniquement)
```bash
python sbert_crossencoder.py
```

### Avec TF-IDF (pour améliorer la détection des compétences spécifiques)
```bash
python sbert_crossencoder.py --use-tfidf
```

### Avec Cross-Encoder (pour une analyse approfondie)
```bash
python sbert_crossencoder.py --use-cross-encoder
```

### Configuration complète (recommandée)
```bash
python sbert_crossencoder.py --use-tfidf --use-cross-encoder
```

### Avec paramètres personnalisés
```bash
python sbert_crossencoder.py --use-tfidf --tfidf-weight 0.4 --use-cross-encoder --cross-encoder-weight 0.8 --top-n 20
```

### Limitation à 2000 offres d'emploi
Le script limite automatiquement le traitement aux 2000 premières offres d'emploi pour optimiser les performances.