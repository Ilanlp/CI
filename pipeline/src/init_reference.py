import pandas as pd
import re


rome40 = pd.read_csv(
    filepath_or_buffer="./data/rome40.csv",
    sep=";",
    header=None,
    names=["A", "B", "C", "D", "E"],
)  # numéro de la colonne qui indexe les entrées

rome40.info()
rome40.describe()
rome40.isna().any(axis=1)
rome40.isnull().sum()
rome40 = rome40.loc[rome40["A"] != " "]
rome40 = rome40.loc[rome40["B"] != " "]
rome40 = rome40.loc[rome40["C"] != " "]
rome40 = rome40.loc[rome40["D"] != " "]
rome40["code"] = (
    rome40["A"].astype(str) + rome40["B"].astype(str) + rome40["C"].astype(str)
)

rome = rome40[["code", "D"]]
rome["libelle"] = rome["D"]
rome.drop("D", axis=1, inplace=True)
rome = rome.drop_duplicates(keep="first")

# rome.head(10)

# print(rome40.head(5))

# DIM_ROMECODES

code_rome = rome["code"]
code_rome = code_rome.drop_duplicates(keep="first")

code_rome.to_csv(
    "./data/DIM_ROMECODES.csv",
    encoding="utf-8",
    header=False,
    index=False,
)

# code_rome.head(10)

# DIM_DOMAINES

domaines = pd.read_csv(
    filepath_or_buffer="./data/france_travail_domaines.csv",
    sep=","
)
domaines.drop(domaines.columns[0], axis=1, inplace=True)
# domaines.info()


domaines.to_csv(
    "./data/DIM_DOMAINES.csv", encoding="utf-8",
    header=False,
    index=False
)

# DIM_APPELLATIONS

appellations = pd.read_csv(
    filepath_or_buffer="./data/france_travail_appellations.csv", sep=","
)
# appellations.info()
# appellations.head()
appellations.drop(appellations.columns[0], axis=1, inplace=True)
# appellations.head()

appellations.to_csv(
    "./data/DIM_APPELLATIONS.csv",
    encoding="utf-8",
    header=False,
    index=False,
)

# DIM_METIERS

metiers = rome.merge(right=appellations, on="libelle", how="left")
metiers.dropna(axis=0, how="all", subset=["code_y"], inplace=True)

columns = {"code_x": "code_rome", "code_y": "code_appellation"}
metiers = metiers.rename(columns, axis=1)

types = {"code_appellation": "int"}
metiers = metiers.astype(types)


def get_first_three_chars(input_string):
    """
    Extrait les 3 premiers caractères d'une chaîne.

    Args:
        input_string (str): La chaîne d'entrée

    Returns:
        str: Les 3 premiers caractères
    """
    pattern = r"^(.{3})"
    match = re.search(pattern, input_string)

    if match:
        return match.group(1)
    else:
        return None


metiers["code_domaine"] = metiers["code_rome"].apply(get_first_three_chars)


def transformer_titre(titre):
    """
    Transforme un titre au format "Masculin / Féminin" en version féminine uniquement.

    Args:
        titre (str): Le titre à transformer

    Returns:
        str: Le titre au féminin
    """
    # Regex pour capturer le schéma "Masculin / Féminin" suivi du reste
    pattern = r"([^\s/]+) / ([^\s/]+)(.*)"

    # Appliquer la transformation
    titre_transforme = re.sub(pattern, r"\1\3", titre)

    return titre_transforme


metiers["libelle_search"] = metiers["libelle"].apply(transformer_titre)

# metiers.head()
# metiers.info()
metiers.to_csv(
    "./data/RAW_METIERS.csv", encoding="utf-8", header=True, index=False
)
