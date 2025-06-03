from pydantic_settings import BaseSettings, SettingsConfigDict
import os, yaml
class Settings(BaseSettings):
    # Chemin du fichier profile.yml
    DBT_PROJECT_NAME: str
    DBT_TARGET: str
    DBT_ACCOUNT: str
    DBT_DATABASE: str
    DBT_PASSWORD: str
    DBT_ROLE: str
    DBT_SCHEMA: str
    DBT_THREADS: str
    DBT_TYPE: str
    DBT_USER: str
    DBT_WAREHOUSE: str
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

settings = Settings()

# Création du fichier profiles.yml
profile_data = {
    settings.DBT_PROJECT_NAME: {
        "outputs": {
            settings.DBT_TARGET: {
                "account": settings.DBT_ACCOUNT,
                "database": settings.DBT_DATABASE,
                "password": settings.DBT_PASSWORD,
                "role": settings.DBT_ROLE,
                "schema": settings.DBT_SCHEMA,
                "threads": int(settings.DBT_THREADS),
                "type": settings.DBT_TYPE,
                "user": settings.DBT_USER,
                "warehouse": settings.DBT_WAREHOUSE
            }
        },
        "target": settings.DBT_TARGET
    }
}

# Dossier de sortie
output_dir = "/home/node/.dbt"

# Écriture dans le fichier profiles.yml
output_path = os.path.join(output_dir, "profiles.yml")
print(output_path)
with open(output_path, "w") as f:
    yaml.dump(profile_data, f, default_flow_style=False)

print(f"Fichier YAML généré : {output_path}")