# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy all files
COPY normalizer.py ./
COPY init_reference.py ./
COPY adzuna/ ./adzuna/
COPY france_travail/ ./france_travail/
COPY /data ./data/        
COPY requirements.txt ./        

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Spécifier la commande de démarrage

CMD ["sh", "-c", "python3 init_reference.py && python3 normalizer.py"]
