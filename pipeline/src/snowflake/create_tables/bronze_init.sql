USE DATABASE JOB_MARKET;
USE SCHEMA RAW;


CREATE TABLE IF NOT EXISTS RAW_CANDIDAT (
  id_candidat INT IDENTITY(1,1) PRIMARY KEY,
  nom VARCHAR(50),
  prenom VARCHAR(50),
  adresse VARCHAR(100),
  email VARCHAR(50),
  tel VARCHAR(25),
  id_competence ARRAY,
  id_softskill ARRAY,
  id_metier ARRAY,
  id_lieu ARRAY,
  id_contrat ARRAY,
  id_type_entreprise ARRAY,
  id_seniorite ARRAY,
  id_teletravail ARRAY,
  id_domaine ARRAY,
  salaire_min  NUMBER
);


CREATE TABLE IF NOT EXISTS RAW_SOFTSKILL (
  code_rome VARCHAR(50),
  uuid VARCHAR(36),
  created_at VARCHAR(100),
  skill_id VARCHAR(40),
  score NUMBER(18,17),
  summary VARCHAR(500),
  details VARCHAR(1000)
);


CREATE TABLE IF NOT EXISTS RAW_ROME_METIER (
  code_rome VARCHAR(50),
  libelle VARCHAR(200),
  code_appellation VARCHAR(100),
  code_domaine VARCHAR(40),
  libelle_search VARCHAR(400)
  
);

CREATE TABLE IF NOT EXISTS RAW_OFFRE (
  id_offre NUMBER AUTOINCREMENT,
  id_local VARCHAR(14),
  source VARCHAR(10000),
  title VARCHAR(1000),
  description VARCHAR(16777216),
  company_name VARCHAR(10000),
  location_name  VARCHAR(10000),
  latitude VARCHAR(200),
  longitude VARCHAR(200),
  date_created TIMESTAMP_LTZ,
  date_updated TIMESTAMP_LTZ,
  contract_type VARCHAR(20),
  contract_duration VARCHAR(50),
  working_hours VARCHAR(14),
  salary_min INT,
  salary_max INT,
  salary_currency VARCHAR(10),
  salary_period VARCHAR(8),
  experience_required VARCHAR(30),
  category VARCHAR(100),
  sector VARCHAR(150),
  application_url VARCHAR(16777216),
  source_url VARCHAR(16777216),
  skills VARCHAR(16777216),
  remote_work VARCHAR(10),
  is_handicap_accessible BOOLEAN,
  code_rome VARCHAR(5),
  langues VARCHAR(100),
  date_extraction TIMESTAMP_LTZ
);
