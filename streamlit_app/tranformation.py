import pandas as pd
import json

with open("offres_adzuna.json", "r", encoding="utf-8") as file:
    data = json.load(file)

offres_adz=[]
for offre in data.get("results"):
    info={
        'id': offre.get("id",''),
        'title': offre.get("title",''),
        'description':offre.get("description",''),
        'created':offre.get("created",''),
        'location':offre.get("location",'').get('display_name',''),
        'latitude':offre.get("latitude",''),
        'longitude':offre.get("longitude",''),
        'company':offre.get("company",'').get('display_name',''),
        'contract_type':offre.get("contract_type",''),
        'salary_is_predicted':offre.get("salary_is_predicted",''),
        'contract_time':offre.get("contract_time",''),
        'redirect_url':offre.get("redirect_url",''),
        'adref':offre.get("adref",''),
        'category':offre.get("category",'').get('label',''),
        'salary_min':offre.get("salary_min",''),
        'salary_max':offre.get("salary_max",''),
        'Ingestion':'Adzuna'
    }
    offres_adz.append(info)





with open("offres_ft.json", "r", encoding="utf-8") as file:
    data = json.load(file)

offres_ft = []

for offre in data.get("resultats", []):
    info = {
        # Identifiants
        "ID": offre.get("id"),
        "Intitulé": offre.get("intitule"),
        "Description": offre.get("description"),
        "Date de publication": offre.get("dateCreation"),
        "Date de mise à jour": offre.get("dateActualisation"),

        # Lieu de travail
        "Lieu - Libellé": offre.get("lieuTravail", {}).get("libelle"),
        "Lieu - Latitude": offre.get("lieuTravail", {}).get("latitude"),
        "Lieu - Longitude": offre.get("lieuTravail", {}).get("longitude"),
        "Lieu - Code postal": offre.get("lieuTravail", {}).get("codePostal"),
        "Lieu - Commune": offre.get("lieuTravail", {}).get("commune"),

        # ROME
        "Code ROME": offre.get("romeCode"),
        "Libellé ROME": offre.get("romeLibelle"),
        "Appellation ROME": offre.get("appellationlibelle"),

        # Entreprise
        "Nom entreprise": offre.get("entreprise", {}).get("nom"),
        "Description entreprise": offre.get("entreprise", {}).get("description"),
        "Logo entreprise": offre.get("entreprise", {}).get("logo"),
        "url entreprise": offre.get("entreprise", {}).get("url"),
        "Acces Handicap entreprise": offre.get("entreprise", {}).get("entrepriseAdaptee"),

        # Contrat
        "Type contrat": offre.get("typeContrat"),
        "Libellé contrat": offre.get("typeContratLibelle"),
        "Nature contrat": offre.get("natureContrat"),

        # Expérience
        "Expérience exigée (code)": offre.get("experienceExige"),
        "Expérience exigée (libellé)": offre.get("experienceLibelle"),
        "Commentaire expérience": offre.get("experienceCommentaire"),

        # Formation (1ère seulement)
        "Formation - Code": offre.get("formations", [{}])[0].get("codeFormation"),
        "Formation - Domaine": offre.get("formations", [{}])[0].get("domaineLibelle"),
        "Formation - Niveau": offre.get("formations", [{}])[0].get("niveauLibelle"),
        "Formation - Commentaire": offre.get("formations", [{}])[0].get("commentaire"),
        "Formation - Exigence": offre.get("formations", [{}])[0].get("exigence"),

        # Langues (1ère)
        "Langue - Libellé": offre.get("langues", [{}])[0].get("libelle"),
        "Langue - Exigence": offre.get("langues", [{}])[0].get("exigence"),

        # Permis (1er)
        "Permis - Libellé": offre.get("permis", [{}])[0].get("libelle"),
        "Permis - Exigence": offre.get("permis", [{}])[0].get("exigence"),

        # Outils bureautiques
        "Outils bureautiques": ", ".join(offre.get("outilsBureautiques", [])),

        # Compétences (1ère)
        "Compétence - Code": offre.get("competences", [{}])[0].get("code"),
        "Compétence - Libellé": offre.get("competences", [{}])[0].get("libelle"),
        "Compétence - Exigence": offre.get("competences", [{}])[0].get("exigence"),

        # Salaire
        "Salaire - Libellé": offre.get("salaire", {}).get("libelle"),
        "Salaire - Commentaire": offre.get("salaire", {}).get("commentaire"),
        "Salaire - Complément 1": offre.get("salaire", {}).get("complement1"),
        "Salaire - Complément 2": offre.get("salaire", {}).get("complement2"),

        # Travail
        "Durée travail": offre.get("dureeTravailLibelle"),
        "Temps plein / partiel": offre.get("dureeTravailLibelleConverti"),
        "Complément exercice": offre.get("complementExercice"),
        "Condition exercice": offre.get("conditionExercice"),

        # Alternance
        "Alternance": offre.get("alternance"),

        # Contact
        "Contact - Nom": offre.get("contact", {}).get("nom"),
        "Contact - Coordonnée 1": offre.get("contact", {}).get("coordonnees1"),
        "Contact - Coordonnée 2": offre.get("contact", {}).get("coordonnees2"),
        "Contact - Coordonnée 3": offre.get("contact", {}).get("coordonnees3"),
        "Contact - Téléphone": offre.get("contact", {}).get("telephone"),
        "Contact - Courriel": offre.get("contact", {}).get("courriel"),
        "Contact - Commentaire": offre.get("contact", {}).get("commentaire"),
        "Contact - URL recruteur": offre.get("contact", {}).get("urlRecruteur"),
        "Contact - URL postulation": offre.get("contact", {}).get("urlPostulation"),

        # Agence
        "Agence - Téléphone": offre.get("agence", {}).get("telephone"),
        "Agence - Courriel": offre.get("agence", {}).get("courriel"),

        # Divers
        "Nombre de postes": offre.get("nombrePostes"),
        "Accessible TH": offre.get("accessibleTH"),
        "Déplacement - Code": offre.get("deplacementCode"),
        "Déplacement - Libellé": offre.get("deplacementLibelle"),
        "Qualification - Code": offre.get("qualificationCode"),
        "Qualification - Libellé": offre.get("qualificationLibelle"),
        "Code NAF": offre.get("codeNAF"),
        "Secteur activité": offre.get("secteurActivite"),
        "Libellé secteur activité": offre.get("secteurActiviteLibelle"),

        # Qualités professionnelles (1ère)
        "Qualité - Libellé": offre.get("qualitesProfessionnelles", [{}])[0].get("libelle"),
        "Qualité - Description": offre.get("qualitesProfessionnelles", [{}])[0].get("description"),

        # Tranche effectif
        "Tranche effectif établissement": offre.get("trancheEffectifEtab"),

        # Origine offre
        "Origine offre - Code": offre.get("origineOffre", {}).get("origine"),
        "Origine offre - URL": offre.get("origineOffre", {}).get("urlOrigine"),

        # Partenaire (1er)
        "Partenaire - Nom": offre.get("origineOffre", {}).get("partenaires", [{}])[0].get("nom"),
        "Partenaire - URL": offre.get("origineOffre", {}).get("partenaires", [{}])[0].get("url"),
        "Partenaire - Logo": offre.get("origineOffre", {}).get("partenaires", [{}])[0].get("logo"),

        # Offre difficile
        "Offre difficile à pourvoir": offre.get("offresManqueCandidats"),

        # Contexte travail
        "Contexte - Horaires": ", ".join(offre.get("contexteTravail", {}).get("horaires", [])),
        "Contexte - Conditions d'exercice": ", ".join(offre.get("contexteTravail", {}).get("conditionsExercice", [])),
    }

    offres_ft.append(info)

df_ft = pd.DataFrame(offres_ft)
df_adzuna =  pd.DataFrame(offres_adz)


df_ft["Ingestion"] = "France Travail"
df_ft["Ingestion"] = "Adzuna"

#df_adzuna.drop(columns = ["Unnamed: 0"],inplace = True)

df_adzuna.rename(columns={
    "id": "ID",
    "title": "Intitulé",
    "description": "Description",
    "created" : "Date de publication",
    "location" : "Lieu - Libellé",
    "latitude" : "Lieu - Latitude",
    "longitude" : "Lieu - Longitude",
    "company" : "Nom entreprise",
    "contract_type	" : "Type contrat",
    "salary_is_predicted" : "Salaire - Libellé",
    "contract_time	" : "Temps plein / partiel",
    "redirect_url" : "Origine offre - URL"
}, inplace=True)

df_concat= pd.concat([df_ft,df_adzuna],ignore_index=True)

print(df_concat)