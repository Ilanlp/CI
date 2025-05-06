# models/src/matched_lieu.py

from snowflake.snowpark.functions import (
    col, lower, expr,
    sin, cos, sqrt, radians, asin,
    lit, when, row_number
)
from snowflake.snowpark.window import Window

# Modèle amélioré :
# - matching géographique (coordonnées)
# - fallback ‘France’ → Paris
# - fallback code département → ville la plus peuplée du code postal

def model(dbt, session):
    raw      = session.table("RAW.RAW_OFFRE")
    dim_lieu = session.table("SILVER.DIM_LIEU")

    # 1. Matching textuel existant
    name_match   = lower(raw["location_name"]).contains(lower(dim_lieu["ville"]))
    dept_match   = lower(raw["location_name"]).contains(lower(dim_lieu["departement"]))
    region_match = lower(raw["location_name"]).contains(lower(dim_lieu["region"]))
    pays_match = lower(raw["location_name"]).contains(lower(dim_lieu["pays"]))

    # 2. Matching géographique (Haversine)
    lat1 = radians(raw["latitude"]);  lon1 = radians(raw["longitude"])
    lat2 = radians(dim_lieu["latitude"]); lon2 = radians(dim_lieu["longitude"])
    dlat = lat1 - lat2; dlon = lon1 - lon2
    a    = sin(dlat/2)**2 + cos(lat2)*cos(lat1)*sin(dlon/2)**2
    c    = 2 * asin(sqrt(a))
    distance_km = lit(6371) * c
    coord_match = distance_km < lit(10)


    # 4. Fallback code département (2 chiffres)
    #dept_code = expr("regexp_substr(location_name, '\\d{2}')")
    dept_code = expr(
    """
    regexp_substr(
      cast(location_name AS string),
      '\\s*([0-9]{2})\\s*',    -- tolère espaces avant/après
      1,                        -- position de départ
      1,                        -- 1ère occurrence
      'e',                      -- mode extraction
      1                         -- renvoie seulement le groupe 1 (les deux chiffres)
    )
    """
)
    dept_code_match = dept_code.is_not_null() & (dim_lieu["code_postal"].substr(1,2) == dept_code)

    # 5. Jointure LEFT avec tous les critères
    join_cond = (
        coord_match
        | pays_match
        | dept_code_match
        | name_match
        | dept_match
        | region_match
    )

    df = (
        raw.join(dim_lieu, join_cond, how="left")
           .select(
               raw["id"].alias("id"),
               raw["id_local"].alias("id_local"),
               raw["location_name"],
               raw["latitude"], raw["longitude"],
               dim_lieu["id_lieu"],
               dim_lieu["ville"].alias("dim_ville"),
               when(coord_match,     1).otherwise(0).alias("is_coord"),
               when(pays_match,    1).otherwise(0).alias("is_france"),
               when(dept_code_match, 1).otherwise(0).alias("is_dept_code"),
               when(name_match,      1).otherwise(0).alias("is_name"),
               when(dept_match,      1).otherwise(0).ali as("is_dept"),
               when(region_match,    1).otherwise(0).alias("is_region"),
               distance_km.alias("distance_km"),
               dim_lieu["population"].alias("population")
           )
    )

    # 6. Fenêtre pour ne garder qu'un seul id_lieu par id_local
    w = Window.partition_by("id_local").order_by(
        col("is_coord").desc(),
        col("distance_km").asc(),
        col("is_name").desc(),
        col("is_dept_code").desc(),
        col("is_dept").desc(),
        col("is_region").desc(),
        col("is_france").desc(),
        col("population").desc()
    )

    best = (
        df
        .with_column("rn", row_number().over(w))
        .filter(col("rn") == 1)
        .drop(
            "is_coord", "is_france", "is_dept_code",
            "is_name", "is_dept", "is_region",
            "distance_km", "population", "rn"
        )
    )

    return best
