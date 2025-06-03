import os
import glob
import logging
from pathlib import Path
import snowflake.connector
from dotenv import load_dotenv
from colorlog import ColoredFormatter
import time

# Configuration du logging
logger = logging.getLogger("alljob")
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
    "../logs/alljob.log", mode="a", encoding="utf-8"
)
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

class SnowflakeLoader:
    def __init__(self, env_file=".env"):
        """Initialise la classe avec les paramètres de connexion à Snowflake."""
        load_dotenv(env_file)
        self.connection = None
        self.cursor = None
        self.filename = None
        self.filepath = None

    def connect(self):
        """Établit la connexion à Snowflake."""
        try:
            self.connection = snowflake.connector.connect(
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA"),
            )
            self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion à Snowflake: {str(e)}")
            return False

    def create_stage(self):
        """Crée un stage temporaire et y télécharge le fichier."""
        try:
            # Créer un stage temporaire avec un nom unique
            stage_name = f"ALL_JOBS_{int(time.time())}"
            self.cursor.execute(f"CREATE TEMPORARY STAGE {stage_name}")
            
            # Télécharger le fichier dans le stage
            self.cursor.execute(f"PUT file://{self.filepath} @{stage_name} AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
            
            self.filename = stage_name
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création du stage: {str(e)}")
            return False

    def drop_stage(self):
        """Supprime le stage temporaire."""
        if self.filename:
            try:
                self.cursor.execute(f"DROP STAGE IF EXISTS {self.filename}")
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du stage: {str(e)}")
        return False

    def load_all_jobs_to_raw_offre(self, database, schema, file_path):
        """Charge le fichier all_jobs.csv.gz dans la table RAW_OFFRE."""
        try:
            # Connexion et préparation
            if not self.connect():
                return False

            self.filepath = os.path.abspath(file_path)
            self.filepath_normalise = self.filepath.replace("\\", "/")
            
            # Configuration de la base et du schéma
            self.cursor.execute(f"USE DATABASE {database}")
            self.cursor.execute(f"USE SCHEMA {schema}")
            
            # Création du format de fichier pour CSV compressé
            format_name = "CSV_GZ_FORMAT"
            self.cursor.execute(f"""
                CREATE FILE FORMAT IF NOT EXISTS {format_name}
                TYPE = CSV
                FIELD_DELIMITER = ','
                SKIP_HEADER = 1
                NULL_IF = ''
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                DATE_FORMAT = AUTO
                TIMESTAMP_FORMAT = AUTO
                COMPRESSION = GZIP
            """)
            
            # Création du stage et chargement du fichier
            if not self.create_stage():
                return False
            
            # Nom complet de la table
            full_table = f"{database}.{schema}.RAW_OFFRE"
            
            # Récupère les colonnes de RAW_OFFRE sauf id_offre
            self.cursor.execute(f"""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'RAW_OFFRE'
                  AND TABLE_SCHEMA = '{schema}'
                  AND COLUMN_NAME != 'ID_OFFRE'
                ORDER BY ORDINAL_POSITION
            """)
            cols = [row[0] for row in self.cursor.fetchall()]
            column_definitions = ", ".join(f'"{col}" VARCHAR' for col in cols)
            
            # Crée une table temporaire avec les bonnes colonnes
            temp_table = f"TEMP_ALL_JOBS_{int(time.time())}"
            self.cursor.execute(f"""
                CREATE TEMPORARY TABLE {temp_table} ({column_definitions})
            """)
            
            # Copie dans la table temporaire
            self.cursor.execute(f"""
                COPY INTO {temp_table}
                FROM @{self.filename}
                FILE_FORMAT = {format_name}
            """)
            
            # Insertion dans la table principale, sans ID_OFFRE
            columns_str = ", ".join(f'"{col}"' for col in cols)
            self.cursor.execute(f"""
                INSERT INTO {full_table} ({columns_str})
                SELECT {columns_str}
                FROM {temp_table}
            """)
            
            # Récupération du nombre de lignes chargées
            rows_inserted = self.cursor.rowcount
            logger.info(f"Chargement réussi dans {full_table}: {rows_inserted} lignes ajoutées")
            
            # Nettoyage
            self.cursor.execute(f"DROP TABLE {temp_table}")
            self.drop_stage()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de all_jobs: {str(e)}")
            return False
        finally:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()

if __name__ == "__main__":
    loader = SnowflakeLoader()
    path_absolu = Path(__file__).resolve()
    output_dir = f"{path_absolu.parents[1]}/data/"

    # Recherche du fichier all_jobs_*.csv.gz le plus récent
    all_jobs_files = glob.glob(f"{output_dir}/all_jobs_*.csv.gz")
    if not all_jobs_files:
        logger.error("Aucun fichier all_jobs_*.csv.gz trouvé dans le dossier de données.")
        exit(1)

    latest_file = max(all_jobs_files, key=os.path.getmtime)
    logger.info(f"Fichier le plus récent détecté : {latest_file}")

    # Chargement du fichier all_jobs.csv.gz le plus récent dans RAW_OFFRE
    success = loader.load_all_jobs_to_raw_offre(
        "JOB_MARKET",
        "RAW",
        latest_file
    )

    if success:
        logger.info("Chargement du fichier all_jobs.csv.gz dans RAW_OFFRE réussi")
    else:
        logger.error("Échec du chargement du fichier all_jobs.csv.gz dans RAW_OFFRE")
        exit(1) 