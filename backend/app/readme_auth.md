# Documentation d'Authentification API

Ce document décrit l'implémentation de l'authentification HTTP Basic pour l'API JobMarket.

## Vue d'ensemble

L'API utilise un système d'authentification HTTP Basic pour sécuriser les opérations de modification des données :
- **Méthodes protégées** : `POST`, `PUT`, `DELETE`, `PATCH`
- **Méthodes publiques** : `GET` (lecture seule)
- **Type d'authentification** : HTTP Basic Authentication

## Architecture

### Structure des fichiers
```
app/
├── auth.py              # Module d'authentification
├── main.py              # Middleware d'authentification
└── api/
    ├── routes.py        # Routes protégées (POST, PUT, DELETE)
    ├── routes_competence.py
    └── ...
```

### Fonctionnement du middleware
```
Requête HTTP → Middleware Auth → Routes → Réponse
                    ↓
            Vérification si méthode 
            nécessite authentification
                    ↓
            Validation des credentials
                    ↓
            Autorisation ou Refus
```

## Configuration

### Credentials par défaut
```python
# app/auth.py
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"  # c'est personnalisable !
```

### Variables d'environnement (recommandé)
Ajoutez dans votre fichier `.env` :
```env
AUTH_USERNAME=your_admin_username
AUTH_PASSWORD=your_secure_password
```

Puis modifiez `app/auth.py` :
```python
import os
from dotenv import load_dotenv

load_dotenv()

ADMIN_USERNAME = os.getenv("AUTH_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("AUTH_PASSWORD", "password123")
```

## Utilisation

### 1. Routes publiques (GET)
Aucune authentification requise :
```bash
curl http://localhost:8000/api/offres
curl http://localhost:8000/api/competences
```

### 2. Routes protégées (POST, PUT, DELETE, PATCH)

#### Avec curl
```bash
# Création d'une offre
curl -X POST http://localhost:8000/api/offres \
  -u admin:password123 \
  -H "Content-Type: application/json" \
  -d '{"title": "Data Engineer", "description": "..."}'

# Mise à jour d'une offre
curl -X PUT http://localhost:8000/api/offres/1 \
  -u admin:password123 \
  -H "Content-Type: application/json" \
  -d '{"title": "Senior Data Engineer"}'

# Suppression d'une offre
curl -X DELETE http://localhost:8000/api/offres/1 \
  -u admin:password123
```

#### Avec Postman
1. Sélectionnez l'onglet **Authorization**
2. Choisissez **Basic Auth**
3. Renseignez :
   - **Username** : `admin`
   - **Password** : `password123`

#### Avec Python requests
```python
import requests
from requests.auth import HTTPBasicAuth

# Authentification
auth = HTTPBasicAuth('admin', 'password123')

# Création d'une ressource
response = requests.post(
    'http://localhost:8000/api/offres',
    json={'title': 'Data Engineer', 'description': '...'},
    auth=auth
)

print(response.status_code)  # 201 si succès
```

#### Avec JavaScript (fetch)
```javascript
// Encoder les credentials en Base64
const credentials = btoa('admin:password123');

fetch('http://localhost:8000/api/offres', {
    method: 'POST',
    headers: {
        'Authorization': `Basic ${credentials}`,
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        title: 'Data Engineer',
        description: '...'
    })
});
```

## 📖 Documentation Swagger/OpenAPI

L'authentification est visible dans la documentation automatique :
- **URL** : `http://localhost:8000/docs`
- **Bouton "Authorize"** en haut à droite
- **Cadenas 🔒** sur les méthodes protégées

### Configuration Swagger
```python
# Dans main.py pour afficher l'auth dans Swagger
from fastapi.security import HTTPBasic

security = HTTPBasic()

@app.post("/api/offres")
async def create_offre(offre: Offre, credentials: HTTPBasicCredentials = Depends(security)):
    # Votre logique...
```

## Codes de réponse

### Succès
- **200** : Opération réussie (PUT, PATCH, DELETE)
- **201** : Ressource créée avec succès (POST)

### Erreurs d'authentification
- **401 Unauthorized** : Credentials manquants ou invalides

```json
{
    "success": false,
    "message": "Authentification requise pour cette opération",
    "data": null
}
```

### Headers de réponse
```
WWW-Authenticate: Basic realm="API"
```

## Dépannage

### Problème : "401 Unauthorized" avec de bons credentials
1. Vérifiez l'encodage Base64 :
   ```bash
   echo -n "admin:password123" | base64
   # Résultat : YWRtaW46cGFzc3dvcmQxMjM=
   ```

2. Testez avec curl :
   ```bash
   curl -H "Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=" \
        -X POST http://localhost:8000/api/offres
   ```

### Problème : L'authentification ne fonctionne pas
1. Vérifiez les logs du serveur pour les messages de debug
2. Assurez-vous que le middleware est bien chargé avant les routes
3. Vérifiez que `app/auth.py` existe et est importé

### Problème : Routes GET protégées par erreur
Le middleware ne protège que les méthodes `POST`, `PUT`, `DELETE`, `PATCH`. Si vos routes GET sont protégées, vérifiez la configuration du middleware.

## Sécurité

### Recommandations de production

1. **Changez les credentials par défaut**
2. **Utilisez HTTPS** en production
3. **Stockez les credentials dans des variables d'environnement**
4. **Considérez l'utilisation de JWT** pour une authentification plus avancée
5. **Implémentez la limitation de taux** (rate limiting)
6. **Loggez les tentatives d'authentification échouées**

### Exemple de configuration sécurisée
```python
# app/auth.py - Version sécurisée
import os
import secrets
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Utiliser des variables d'environnement
ADMIN_USERNAME = os.getenv("AUTH_USERNAME")
ADMIN_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH")  # Hash du mot de passe

def verify_password(password: str, hashed_password: str) -> bool:
    """Vérification sécurisée avec hash"""
    return secrets.compare_digest(
        hashlib.sha256(password.encode()).hexdigest(),
        hashed_password
    )
```

## Support

Pour toute question concernant l'authentification :
- **Email** : support@jobmarket.com # (à modifier)
- **Documentation API** : http://localhost:8000/docs
- **Logs** : Consultez les logs du serveur pour les détails des erreurs

---