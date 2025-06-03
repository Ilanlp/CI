import os
import glob
import re
import logging
from pathlib import Path
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
from dotenv import load_dotenv
from colorlog import ColoredFormatter


logger = logging.getLogger("init")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler(
    "../logs/snowflake_core.log", mode="a", encoding="utf-8"
)
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class SnowflakeCore:
    """
    Classe pour charger des données CSV dans Snowflake et effectuer des insertions
    dans différentes tables en gérant les relations entre elles.
    """

    def __init__(self, env_file=".env"):
        """
        Initialise la classe avec les paramètres de connexion à Snowflake.

        Args:
            env_file (str): Chemin vers le fichier .env contenant les identifiants
        """
        # Charge les variables d'environnement depuis le fichier .env
        load_dotenv(env_file)
        # Initialisation des attributs de classe
        self.connection = None  # Connexion à Snowflake
        self.cursor = None  # Curseur pour exécuter les requêtes
        self.filename = None  # Nom du fichier CSV traité
        self.filepath = None  # Chemin complet du fichier CSV

    def connect(self):
        """
        Établit la connexion à Snowflake en utilisant les identifiants
        stockés dans les variables d'environnement.

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes
        """
        self.connection = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
        )
        self.cursor = self.connection.cursor()
        return self

    def is_connected(self):
        """
        Vérifie si la connexion à Snowflake est active et fonctionnelle
        en exécutant une requête simple.

        Returns:
            bool: True si la connexion est établie et fonctionnelle, False sinon
        """
        if self.connection is None or self.cursor is None:
            return False

        try:
            # Exécute une requête simple pour vérifier la connexion
            self.cursor.execute("SELECT 1")
            result = self.cursor.fetchone()
            return result is not None and result[0] == 1
        except Exception as e:
            print(f"Erreur lors de la vérification de la connexion: {str(e)}")
            return False

    def create_stage(self):
        """
        Crée un stage Snowflake pour le téléchargement du fichier CSV.
        Le stage est un espace de stockage temporaire sur Snowflake.

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes
        """
        if self.is_connected():
            # Crée le stage s'il n'existe pas déjà
            self.cursor.execute(f"CREATE STAGE IF NOT EXISTS {self.filename}")
            # Télécharge le fichier dans le stage
            self.cursor.execute(
                f"PUT file://{self.filepath_normalise} @{self.filename} auto_compress=true"
            )
            return self

        return False

    def drop_stage(self, stage_name=None):
        """
        Supprime un stage Snowflake.

        Args:
            stage_name (str): Nom du stage à supprimer (utilise self.filename si None)

        Returns:
            bool: True si la suppression réussit, False sinon
        """
        if not self.is_connected():
            return False

        try:
            # Utiliser le nom du stage par défaut si non spécifié
            stage_name = stage_name or self.filename

            # Exécuter la commande DROP STAGE
            self.cursor.execute(f"DROP STAGE IF EXISTS {stage_name}")
            print(f"Stage {stage_name} supprimé avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors de la suppression du stage {stage_name}: {str(e)}")
            return False

    def create_file_format(
        self, format_name, format_type="CSV", schema_name=None, options=None
    ):
        """
        Crée un file format dans Snowflake pour l'import de données.

        Args:
            format_name (str): Nom du format à créer
            format_type (str): Type de format (CSV, JSON, PARQUET, etc.)
            schema_name (str): Nom du schéma où créer le format (utilise le schéma courant si None)
            options (dict): Options spécifiques au format

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes

        Returns:
            tuple: (schema_name, format_name) pour référence future
        """
        if not self.is_connected():
            return False

        # Options par défaut pour CSV si non spécifiées
        if options is None and format_type.upper() == "CSV":
            options = {
                "FIELD_DELIMITER": ",",
                "SKIP_HEADER": 1,
                "NULL_IF": "''",
                "FIELD_OPTIONALLY_ENCLOSED_BY": '"',
                "DATE_FORMAT": "AUTO",
                "TIMESTAMP_FORMAT": "AUTO",
            }

        # Construire la clause OPTIONS
        options_clause = ""
        if options:
            options_parts = []
            for key, value in options.items():
                if isinstance(value, str):
                    options_parts.append(f"{key} = '{value}'")
                else:
                    options_parts.append(f"{key} = {value}")
            options_clause = "(" + ", ".join(options_parts) + ")"

        # Utiliser le schéma courant si non spécifié
        if not schema_name:
            # Récupérer le schéma courant
            self.cursor.execute("SELECT CURRENT_SCHEMA()")
            schema_name = self.cursor.fetchone()[0]

        # Créer le nom complet du format avec le schéma
        qualified_format_name = f"{schema_name}.{format_name}"

        # Exécuter la requête de création
        self.cursor.execute(
            f"""
            CREATE FILE FORMAT IF NOT EXISTS {qualified_format_name}
            TYPE = {format_type.upper()}
            {options_clause}
            """
        )

        logger.info(f"Format de fichier {qualified_format_name} créé avec succès")
        return (schema_name, format_name)

    def copy_from_stage(
        self,
        table_name,
        stage_name=None,
        schema_name=None,
        file_format=None,
        file_format_schema=None,
        pattern=None,
    ):
        """
        Copie des données d'un stage Snowflake vers une table.

        Args:
            table_name (str): Nom de la table cible
            stage_name (str): Nom du stage source (utilise self.filename si None)
            schema_name (str): Nom du schéma contenant la table
            file_format (str): Nom du format de fichier à utiliser
            file_format_schema (str): Nom du schéma contenant le format de fichier
            pattern (str): Motif pour filtrer les fichiers dans le stage

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes
        """
        if not self.is_connected():
            return False

        # Utiliser les valeurs par défaut si non spécifiées
        stage_name = stage_name or self.filename

        # Qualifier correctement le nom de la table avec son schéma
        qualified_table_name = table_name
        if schema_name:
            qualified_table_name = f"{schema_name}.{table_name}"

        # Qualifier correctement le nom du format de fichier avec son schéma
        qualified_format_name = file_format
        if file_format and file_format_schema:
            qualified_format_name = f"{file_format_schema}.{file_format}"

        # Construire les clauses conditionnelles
        format_clause = f"FILE_FORMAT = {qualified_format_name}" if file_format else ""
        pattern_clause = f"PATTERN = '{pattern}'" if pattern else ""

        # Construire et exécuter la requête COPY
        copy_query = f"""
        COPY INTO {qualified_table_name}
        FROM @{stage_name}
        {format_clause}
        {pattern_clause}
        """

        self.cursor.execute(copy_query)
        result = self.cursor.fetchall()

        # Journaliser le résultat
        rows_loaded = sum(row[2] for row in result) if result else 0
        logger.info(
            f"Données copiées du stage {stage_name} vers la table {qualified_table_name}: {rows_loaded} lignes"
        )
        print(f"Données copiées avec succès: {rows_loaded} lignes")

        return self

    def load_file(self, directory_path):
        """
        Charge un fichier CSV commençant par 'all_jobs' dans le répertoire spécifié.
        Utilise une expression régulière pour trouver les fichiers correspondants.

        Args:
            directory_path (str): Chemin du répertoire contenant les fichiers

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes

        Raises:
            FileNotFoundError: Si aucun fichier correspondant n'est trouvé
        """
        # Récupérer tous les fichiers CSV du répertoire
        all_files = glob.glob(os.path.join(directory_path, "*.csv"))

        # Utiliser regex pour trouver le fichier qui commence par all_jobs
        pattern = re.compile(r"all_jobs.*\.csv$")
        matched_files = [f for f in all_files if pattern.search(os.path.basename(f))]

        # Lever une exception si aucun fichier ne correspond
        if not matched_files:
            raise FileNotFoundError(
                "Aucun fichier commençant par 'all_jobs' n'a été trouvé."
            )

        # Utiliser le premier fichier correspondant
        self.filepath = os.path.abspath(matched_files[0])
        # Normaliser le chemin pour Snowflake (utiliser des / au lieu de \)
        self.filepath_normalise = self.filepath.replace("\\", "/")
        # Extraire le nom du fichier sans extension et le mettre en majuscules
        self.filename = os.path.basename(self.filepath).split(".")[0].upper()
        print(f"Préparation du fichier: {self.filename}")
        return self

    def load_to_table(self, chunk_size=10000):
        """
        Charge les données du fichier CSV dans une table temporaire Snowflake.
        Utilise pandas pour lire le CSV et snowflake.connector pour le chargement.

        Args:
            chunk_size (int): Nombre de lignes à charger par lot pour optimiser la mémoire

        Returns:
            self: Retourne l'instance pour permettre le chaînage des méthodes
        """
        if self.is_connected():
            # Lire le fichier CSV avec pandas
            df = pd.read_csv(self.filepath)
            # Utiliser write_pandas pour charger les données dans une table temporaire
            success, nchunks, nrows, output = write_pandas(
                self.connection,
                df,
                self.filename,  # Nom de la table temporaire
                auto_create_table=True,  # Créer la table automatiquement
                table_type="temporary",  # Table temporaire
                chunk_size=chunk_size,  # Taille des lots
            )
            print(f"Succès: {success}")
            print(f"Chunks utilisés: {nchunks}")
            print(f"Lignes insérées: {nrows}")

            # Supprimer le fichier local après le chargement
            # os.remove(self.filepath)
            return self

        return False

    def insert_all_job(self):
        """
        Insère les données depuis la table temporaire vers les tables permanentes.
        Gère les relations entre les tables ENTREPRISE et OFFREEMPLOI.

        Returns:
            bool: True si l'insertion réussit, False sinon
        """
        if self.is_connected():
            # Insertion des entreprises qui n'existent pas encore
            self.cursor.execute(
                f"""
            INSERT INTO ENTREPRISE (nom)
            SELECT DISTINCT t."company_name" FROM {self.filename} t
            WHERE NOT EXISTS (
                SELECT 1 FROM ENTREPRISE e
                WHERE e.nom = t."company_name"
            )
            """
            )

            # Insertion des offres d'emploi en reliant à l'entreprise correspondante
            self.cursor.execute(
                f"""
            INSERT INTO OFFREEMPLOI (id, description, date_publication, date_mise_a_jour, source, source_url, salaire, id_entreprise)
            SELECT
                o."id",
                o."description",
                o."date_created",
                o."date_updated",
                o."source",
                o."source_url",
                o."salary_min",
                e.id_entreprise
            FROM {self.filename} o
            LEFT JOIN ENTREPRISE e ON o."company_name" = e.nom
            """
            )

            return True

        return False

    def drop_db(self):
        """
        Supprime la table temporaire créée lors du chargement.

        Returns:
            bool: True si la suppression réussit, False sinon
        """
        if self.is_connected():
            self.cursor.execute(
                f"""
            DROP TABLE IF EXISTS {self.filename}
            """
            )
            return True
        return False

    def process_file(self, directory_path, chunk_size=10000):
        """
        Méthode principale pour traiter un fichier de bout en bout.
        Enchaîne toutes les étapes du processus de chargement.

        Args:
            directory_path (str): Chemin du répertoire contenant les fichiers CSV
            chunk_size (int): Taille des chunks pour le chargement

        Returns:
            bool: True si le traitement réussit, False sinon
        """
        try:
            # Établir la connexion et enchaîner les opérations
            self.connect()
            (
                self.load_file(directory_path)
                .create_stage()
                .load_to_table(chunk_size)
                .insert_all_job()
            )
            return True
        except Exception as e:
            # Capturer et afficher les erreurs
            print(f"Erreur lors du traitement du fichier: {str(e)}")
            return False
        finally:
            # Fermer la connexion et le curseur dans tous les cas
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()

    def detect_file_type(self, file_path):
        """
        Détecte le type de fichier en fonction de son extension.

        Args:
            file_path (str): Chemin du fichier

        Returns:
            str: Type de fichier ('CSV' ou 'PARQUET')
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension == ".csv":
            return "CSV"
        elif extension in (".parquet", ".pq"):
            return "PARQUET"
        else:
            raise ValueError(f"Type de fichier non pris en charge: {extension}")

    def create_file_format_for_type(
        self, format_name, file_type, schema_name=None, options=None
    ):
        """
        Crée un format de fichier adapté au type de fichier détecté.

        Args:
            format_name (str): Nom du format à créer
            file_type (str): Type de fichier ('CSV' ou 'PARQUET')
            schema_name (str): Nom du schéma où créer le format
            options (dict): Options spécifiques au format (remplace les options par défaut)

        Returns:
            tuple: (schema_name, format_name) pour référence future
        """
        # Options par défaut selon le type de fichier
        default_options = {}

        if file_type == "CSV":
            default_options = {
                "FIELD_DELIMITER": ",",
                "SKIP_HEADER": 1,
                "NULL_IF": "''",
                "FIELD_OPTIONALLY_ENCLOSED_BY": '"',
                "DATE_FORMAT": "AUTO",
                "TIMESTAMP_FORMAT": "AUTO",
            }
        elif file_type == "PARQUET":
            # Les fichiers Parquet ont généralement peu d'options car le format est auto-descriptif
            default_options = {}

        # Utiliser les options fournies ou les options par défaut
        final_options = default_options
        if options:
            final_options.update(options)

        # Créer le format de fichier
        return self.create_file_format(
            format_name=format_name,
            format_type=file_type,
            schema_name=schema_name,
            options=final_options,
        )

    def process_file_with_copy(
        self,
        database,
        schema_name,
        table_name,
        file_path,
        file_format_name=None,
        file_format_schema=None,
        format_options=None,
        use_pyarrow=False,
        skip_if_data_exists=True,
        drop_stage_after=True,
    ):
        """
        Méthode pour charger un fichier (CSV ou Parquet) dans Snowflake.

        Args:
            database (str): Nom de la base de données cible
            schema_name (str): Nom du schéma de la table cible
            table_name (str): Nom de la table cible
            file_path (str): Chemin complet du fichier à charger
            file_format_name (str): Nom du format de fichier à utiliser
            file_format_schema (str): Nom du schéma contenant le format de fichier
            format_options (dict): Options spécifiques pour le format de fichier
            use_pyarrow (bool): Si True, utilise PyArrow pour les fichiers Parquet
            skip_if_data_exists (bool): Si True, ignore l'import si la table existe et contient des données

        Returns:
            bool: True si le traitement réussit ou si la table est ignorée, False en cas d'erreur
        """
        try:
            # Établir la connexion
            self.connect()

            # Extraire et normaliser le chemin du fichier
            self.filepath = os.path.abspath(file_path)
            self.filepath_normalise = self.filepath.replace("\\", "/")

            # Extraire le nom du fichier sans extension et le mettre en majuscules pour le stage
            self.filename = os.path.basename(self.filepath).split(".")[0].upper()
            print(f"Préparation du fichier: {self.filename}")

            # Vérifier si la table existe et contient des données
            if skip_if_data_exists and self.table_exists_and_has_data(
                database, schema_name, table_name
            ):
                print(
                    f"La table {database}.{schema_name}.{table_name} existe déjà et contient des données. Import ignoré."
                )
                return True  # Retourner True car ce n'est pas une erreur

            # Détecter le type de fichier
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # Traitement selon le type de fichier
            result = False

            # Traitement selon le type de fichier
            if extension == ".csv":
                # Traitement d'un fichier CSV
                result = self.process_csv_file_with_copy(
                    database=database,
                    schema_name=schema_name,
                    table_name=table_name,
                    file_path=file_path,
                    file_format_name=file_format_name,
                    file_format_schema=file_format_schema,
                    format_options=format_options,
                )
            elif extension in (".parquet", ".pq"):
                # Traitement d'un fichier Parquet
                if use_pyarrow:
                    result = self.process_parquet_file_with_pyarrow(
                        database=database,
                        schema_name=schema_name,
                        table_name=table_name,
                        file_path=file_path,
                        file_format_name=file_format_name,
                        file_format_schema=file_format_schema,
                        format_options=format_options,
                    )
                else:
                    result = self.process_parquet_file_with_variant(
                        database=database,
                        schema_name=schema_name,
                        table_name=table_name,
                        file_path=file_path,
                        file_format_name=file_format_name,
                        file_format_schema=file_format_schema,
                        format_options=format_options,
                    )
            else:
                raise ValueError(f"Type de fichier non pris en charge: {extension}")

            # Supprimer le stage si demandé et si le chargement a réussi
            if result and drop_stage_after:
                self.drop_stage()

            return result

        except Exception as e:
            # Capturer et afficher les erreurs
            print(f"Erreur lors du traitement du fichier: {str(e)}")
            return False
        finally:
            # Fermer la connexion et le curseur dans tous les cas
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()

    def process_csv_file_with_copy(
        self,
        database,
        schema_name,
        table_name,
        file_path,
        file_format_name=None,
        file_format_schema=None,
        format_options=None,
    ):
        """
        Méthode pour charger un fichier CSV dans Snowflake.

        Args:
            database (str): Nom de la base de données cible
            schema_name (str): Nom du schéma de la table cible
            table_name (str): Nom de la table cible
            file_path (str): Chemin complet du fichier CSV à charger
            file_format_name (str): Nom du format de fichier à utiliser
            file_format_schema (str): Nom du schéma contenant le format de fichier
            format_options (dict): Options spécifiques pour le format de fichier

        Returns:
            bool: True si le traitement réussit, False en cas d'erreur
        """
        try:
            # Utiliser la base de données spécifiée
            self.cursor.execute(f"USE DATABASE {database}")

            # Utiliser le schéma spécifié
            self.cursor.execute(f"USE SCHEMA {schema_name}")

            # Si aucun format de fichier n'est spécifié, en créer un pour CSV
            if not file_format_name:
                file_format_name = "CSV_FORMAT"
                format_schema, format_name = self.create_file_format(
                    format_name=file_format_name,
                    format_type="CSV",
                    schema_name=schema_name,
                    options=format_options,
                )
                # Mettre à jour le schéma du format de fichier pour la commande COPY
                file_format_schema = format_schema

            # Créer le stage et télécharger le fichier
            self.create_stage()

            # Qualifier complètement le nom de la table et du format
            fully_qualified_table = f"{database}.{schema_name}.{table_name}"
            qualified_format = (
                f"{file_format_schema}.{file_format_name}"
                if file_format_schema
                else file_format_name
            )

            # Copier les données du stage vers la table cible
            self.cursor.execute(
                f"""
            COPY INTO {fully_qualified_table}
            FROM @{self.filename}
            FILE_FORMAT = {qualified_format}
            """
            )

            result = self.cursor.fetchall()
            rows_loaded = sum(row[2] for row in result) if result else 0

            print(
                f"Fichier CSV {self.filename} chargé avec succès dans {fully_qualified_table}: {rows_loaded} lignes"
            )
            return True

        except Exception as e:
            print(f"Erreur lors du traitement du fichier CSV: {str(e)}")
            return False

    def process_parquet_file_with_pyarrow(
        self,
        database,
        schema_name,
        table_name,
        file_path,
        file_format_name=None,
        file_format_schema=None,
        format_options=None,
    ):
        """
        Méthode pour charger un fichier Parquet dans Snowflake en utilisant PyArrow
        pour analyser le schéma.

        Args:
            database (str): Nom de la base de données cible
            schema_name (str): Nom du schéma de la table cible
            table_name (str): Nom de la table cible
            file_path (str): Chemin complet du fichier Parquet à charger
            file_format_name (str): Nom du format de fichier à utiliser
            file_format_schema (str): Nom du schéma contenant le format de fichier
            format_options (dict): Options spécifiques pour le format de fichier

        Returns:
            bool: True si le traitement réussit, False en cas d'erreur
        """
        try:
            # Vérifier que c'est bien un fichier Parquet
            if not file_path.lower().endswith((".parquet", ".pq")):
                raise ValueError("Le fichier doit être au format Parquet.")

            # Importer pyarrow
            import pyarrow.parquet as pq

            # Lire le schéma du fichier Parquet
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema

            # Utiliser la base de données et le schéma spécifiés
            self.cursor.execute(f"USE DATABASE {database}")
            self.cursor.execute(f"USE SCHEMA {schema_name}")

            # Mapper les types PyArrow vers les types Snowflake
            snowflake_columns = []
            for field in schema.names:
                arrow_type = schema.field(field).type
                # Conversion des types PyArrow en types Snowflake
                if "int" in str(arrow_type).lower():
                    sf_type = "NUMBER"
                elif "float" in str(arrow_type).lower():
                    sf_type = "FLOAT"
                elif "date" in str(arrow_type).lower():
                    sf_type = "DATE"
                elif "timestamp" in str(arrow_type).lower():
                    sf_type = "TIMESTAMP"
                elif (
                    "string" in str(arrow_type).lower()
                    or "bytes" in str(arrow_type).lower()
                ):
                    sf_type = "VARCHAR"
                elif "bool" in str(arrow_type).lower():
                    sf_type = "BOOLEAN"
                else:
                    sf_type = "VARCHAR"  # Type par défaut

                snowflake_columns.append(f'"{field}" {sf_type}')

            # Créer la table si elle n'existe pas
            fully_qualified_table = f"{database}.{schema_name}.{table_name}"
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {fully_qualified_table} (
                {', '.join(snowflake_columns)}
            )
            """
            self.cursor.execute(create_table_query)
            print(
                f"Table {fully_qualified_table} créée ou vérifiée avec succès via PyArrow"
            )

            # Créer le format de fichier pour Parquet si nécessaire
            if not file_format_name:
                file_format_name = "PARQUET_FORMAT"
                format_schema, format_name = self.create_file_format(
                    format_name=file_format_name,
                    format_type="PARQUET",
                    schema_name=schema_name,
                    options=format_options,
                )
                file_format_schema = format_schema

            # Créer le stage et télécharger le fichier
            self.create_stage()

            # Qualifier complètement le nom du format
            qualified_format = (
                f"{file_format_schema}.{file_format_name}"
                if file_format_schema
                else file_format_name
            )

            # Exécuter COPY INTO pour charger les données
            self.cursor.execute(
                f"""
            COPY INTO {fully_qualified_table}
            FROM @{self.filename}
            FILE_FORMAT = {qualified_format}
            """
            )

            result = self.cursor.fetchall()
            rows_loaded = sum(row[2] for row in result) if result else 0

            print(
                f"Fichier Parquet chargé avec succès via PyArrow: {rows_loaded} lignes"
            )
            return True

        except ImportError:
            print(
                "PyArrow n'est pas installé. Veuillez l'installer avec 'pip install pyarrow'."
            )
            return False
        except Exception as e:
            print(
                f"Erreur lors du traitement du fichier Parquet avec PyArrow: {str(e)}"
            )
            return False

    def process_parquet_file_with_variant(
        self,
        database,
        schema_name,
        table_name,
        file_path,
        file_format_name=None,
        file_format_schema=None,
        format_options=None,
    ):
        """
        Méthode pour charger un fichier Parquet dans Snowflake en utilisant une table temporaire
        avec une colonne VARIANT pour inférer le schéma.

        Args:
            database (str): Nom de la base de données cible
            schema_name (str): Nom du schéma de la table cible
            table_name (str): Nom de la table cible
            file_path (str): Chemin complet du fichier Parquet à charger
            file_format_name (str): Nom du format de fichier à utiliser
            file_format_schema (str): Nom du schéma contenant le format de fichier
            format_options (dict): Options spécifiques pour le format de fichier

        Returns:
            bool: True si le traitement réussit, False en cas d'erreur
        """
        try:
            # Vérifier que c'est bien un fichier Parquet
            if not file_path.lower().endswith((".parquet", ".pq")):
                raise ValueError("Le fichier doit être au format Parquet.")

            # Import pour le timestamp dans le nom de la table temporaire
            import time

            # Utiliser la base de données et le schéma spécifiés
            self.cursor.execute(f"USE DATABASE {database}")
            self.cursor.execute(f"USE SCHEMA {schema_name}")

            # Créer un format de fichier Parquet si nécessaire
            if not file_format_name:
                file_format_name = "PARQUET_FORMAT"
                format_schema, format_name = self.create_file_format(
                    format_name=file_format_name,
                    format_type="PARQUET",
                    schema_name=schema_name,
                    options=format_options,
                )
                file_format_schema = format_schema

            # Créer le stage et télécharger le fichier
            self.create_stage()

            # Qualifier complètement le nom du format
            qualified_format = (
                f"{file_format_schema}.{file_format_name}"
                if file_format_schema
                else file_format_name
            )

            # Créer une table temporaire avec une seule colonne VARIANT
            temp_table_name = f"TEMP_{self.filename}_{int(time.time())}"
            self.cursor.execute(
                f"""
            CREATE TEMPORARY TABLE {temp_table_name} (
                parquet_data VARIANT
            )
            """
            )

            # Charger les données dans la table temporaire
            self.cursor.execute(
                f"""
            COPY INTO {temp_table_name} (parquet_data)
            FROM @{self.filename}
            FILE_FORMAT = {qualified_format}
            """
            )

            # Vérifier si la table cible existe déjà
            self.cursor.execute(
                f"""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_CATALOG = '{database}'
            AND TABLE_SCHEMA = '{schema_name}'
            AND TABLE_NAME = '{table_name}'
            """
            )
            table_exists = self.cursor.fetchone()[0] > 0

            if not table_exists:
                # Analyser quelques lignes pour inférer le schéma
                self.cursor.execute(f"SELECT * FROM {temp_table_name} LIMIT 1")
                sample_data = self.cursor.fetchone()

                if sample_data and sample_data[0]:
                    # Exécuter une requête pour obtenir la structure des données
                    self.cursor.execute(
                        f"""
                    SELECT DISTINCT
                        key as column_name,
                        typeof(value) as data_type
                    FROM {temp_table_name},
                    LATERAL FLATTEN(INPUT => parquet_data)
                    ORDER BY column_name
                    """
                    )

                    column_types = self.cursor.fetchall()

                    # Créer la définition des colonnes
                    columns = []
                    for col_name, col_type in column_types:
                        # Mapper les types JSON vers les types Snowflake
                        if col_type.upper() == "NUMBER":
                            sf_type = "NUMBER"
                        elif col_type.upper() in ("FLOAT", "DOUBLE"):
                            sf_type = "FLOAT"
                        elif col_type.upper() == "BOOLEAN":
                            sf_type = "BOOLEAN"
                        elif col_type.upper() == "ARRAY":
                            sf_type = "ARRAY"
                        elif col_type.upper() == "OBJECT":
                            sf_type = "OBJECT"
                        else:
                            sf_type = "VARCHAR"

                        columns.append(f'"{col_name}" {sf_type}')

                    # Créer la table cible avec le schéma inféré
                    fully_qualified_table = f"{database}.{schema_name}.{table_name}"
                    if columns:
                        create_table_query = f"""
                        CREATE TABLE {fully_qualified_table} (
                            {', '.join(columns)}
                        )
                        """
                        self.cursor.execute(create_table_query)
                        print(
                            f"Table {fully_qualified_table} créée avec succès via la méthode VARIANT"
                        )
                    else:
                        raise ValueError(
                            "Impossible d'inférer le schéma du fichier Parquet"
                        )
                else:
                    raise ValueError("Le fichier Parquet semble être vide")

            # Insérer les données de la table temporaire dans la table cible
            fully_qualified_table = f"{database}.{schema_name}.{table_name}"

            # Construire dynamiquement la requête d'insertion en fonction des colonnes
            self.cursor.execute(
                f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_CATALOG = '{database}' 
            AND TABLE_SCHEMA = '{schema_name}' 
            AND TABLE_NAME = '{table_name}'
            """
            )

            columns = [row[0] for row in self.cursor.fetchall()]
            columns_str = ", ".join([f'"{col}"' for col in columns])
            path_str = ", ".join([f'parquet_data:"{col}"' for col in columns])

            # Insérer les données
            self.cursor.execute(
                f"""
            INSERT INTO {fully_qualified_table} ({columns_str})
            SELECT {path_str}
            FROM {temp_table_name}
            """
            )

            rows_inserted = self.cursor.rowcount
            print(f"{rows_inserted} lignes insérées dans {fully_qualified_table}")

            # Supprimer la table temporaire
            self.cursor.execute(f"DROP TABLE {temp_table_name}")

            return True
        except Exception as e:
            print(
                f"Erreur lors du traitement du fichier Parquet avec la méthode VARIANT: {str(e)}"
            )
            return False

    def table_exists_and_has_data(self, database, schema_name, table_name):
        """
        Vérifie si une table existe et contient des données.

        Args:
            database (str): Nom de la base de données
            schema_name (str): Nom du schéma
            table_name (str): Nom de la table

        Returns:
            bool: True si la table existe et contient des données, False sinon
        """
        if not self.is_connected():
            return False

        try:
            # Vérifier si la table existe
            self.cursor.execute(
                f"""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_CATALOG = '{database}' 
            AND TABLE_SCHEMA = '{schema_name}' 
            AND TABLE_NAME = '{table_name}'
            """
            )

            table_exists = self.cursor.fetchone()[0] > 0

            if not table_exists:
                return False

            # Si la table existe, vérifier si elle contient des données
            self.cursor.execute(
                f"""
            SELECT COUNT(*) FROM {database}.{schema_name}.{table_name} LIMIT 1
            """
            )

            row_count = self.cursor.fetchone()[0]
            return row_count > 0

        except Exception as e:
            print(
                f"Erreur lors de la vérification de la table {database}.{schema_name}.{table_name}: {str(e)}"
            )
            return False

def load_table_parallel(loader, database, schema, table, file_path, file_format_name=None, file_format_schema=None):
    """
    Fonction pour charger une table en parallèle.
    Crée une nouvelle instance de SnowflakeCore pour chaque chargement.
    """
    try:
        # Créer une nouvelle instance pour chaque thread
        thread_loader = SnowflakeCore()
        success = thread_loader.process_file_with_copy(
            database,
            schema,
            table,
            file_path,
            file_format_name,
            file_format_schema
        )
        if success:
            logging.info(f"Traitement du fichier {table} terminé avec succès")
        else:
            logging.error(f"Échec du traitement du fichier {table}")
        return success
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {table}: {str(e)}")
        return False

if __name__ == "__main__":
    import concurrent.futures
    from pathlib import Path

    # Configuration des tâches de chargement
    path_absolu = Path(__file__).resolve()
    output_dir = f"{path_absolu.parents[1]}/data/"

    # Liste des tâches de chargement à exécuter en parallèle
    loading_tasks = [
        # RAW Layer
        ("JOB_MARKET", "RAW", "RAW_CANDIDAT", f"{output_dir}/RAW_CANDIDAT.csv", "CSV_ERROR", "PUBLIC"),
        ("JOB_MARKET", "RAW", "RAW_SOFTSKILL", f"{output_dir}/RAW_SOFTSKILL.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "RAW", "RAW_ROME_METIER", f"{output_dir}/RAW_ROME_METIER.csv", "CLASSIC_CSV", "PUBLIC"),
        
        # SILVER Layer
        ("JOB_MARKET", "SILVER", "DIM_STOPWORDS", f"{output_dir}/DIM_STOPWORDS.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_ENTREPRISE", f"{output_dir}/DIM_ENTREPRISE.parquet", None, None),
        ("JOB_MARKET", "SILVER", "DIM_COMPETENCE", f"{output_dir}/DIM_COMPETENCE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_CONTRAT", f"{output_dir}/DIM_CONTRAT.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_DATE", f"{output_dir}/DIM_DATE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_DOMAINE", f"{output_dir}/DIM_DOMAINE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_LIEU", f"{output_dir}/DIM_LIEU.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_METIER", f"{output_dir}/DIM_METIER.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_ROMECODE", f"{output_dir}/DIM_ROMECODE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_SENIORITE", f"{output_dir}/DIM_SENIORITE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_SOFTSKILL", f"{output_dir}/DIM_SOFTSKILL.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "SILVER", "DIM_TELETRAVAIL", f"{output_dir}/DIM_TELETRAVAIL.csv", "CLASSIC_CSV", "PUBLIC"),
        
        
        # GOLD Layer
        ("JOB_MARKET", "GOLD", "DIM_ENTREPRISE", f"{output_dir}/DIM_ENTREPRISE.parquet", None, None),
        ("JOB_MARKET", "GOLD", "DIM_COMPETENCE", f"{output_dir}/DIM_COMPETENCE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_CONTRAT", f"{output_dir}/DIM_CONTRAT.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_DATE", f"{output_dir}/DIM_DATE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_DOMAINE", f"{output_dir}/DIM_DOMAINE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_LIEU", f"{output_dir}/DIM_LIEU.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_METIER", f"{output_dir}/DIM_METIER.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_ROMECODE", f"{output_dir}/DIM_ROMECODE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_SENIORITE", f"{output_dir}/DIM_SENIORITE.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_SOFTSKILL", f"{output_dir}/DIM_SOFTSKILL.csv", "CLASSIC_CSV", "PUBLIC"),
        ("JOB_MARKET", "GOLD", "DIM_TELETRAVAIL", f"{output_dir}/DIM_TELETRAVAIL.csv", "CLASSIC_CSV", "PUBLIC"),
       
    ]

    # Créer une instance de base pour les tests de connexion
    base_loader = SnowflakeCore()
    
    # Vérifier la connexion avant de commencer
    if not base_loader.connect():
        logging.error("Impossible de se connecter à Snowflake. Arrêt du programme.")
        exit(1)

    # Exécuter les chargements en parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Soumettre toutes les tâches
        future_to_task = {
            executor.submit(load_table_parallel, base_loader, *task): task 
            for task in loading_tasks
        }
        
        # Attendre la fin de toutes les tâches et collecter les résultats
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                success = future.result()
                if success:
                    logging.info(f"Tâche {task[2]} terminée avec succès")
                else:
                    logging.error(f"Échec de la tâche {task[2]}")
            except Exception as e:
                logging.error(f"Erreur lors de l'exécution de la tâche {task[2]}: {str(e)}")

    logging.info("Tous les chargements sont terminés")