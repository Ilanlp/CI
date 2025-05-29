import secrets
import base64

# Configuration de l'authentification basique
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"  # Changez ce mot de passe !

def verify_basic_auth(auth_header: str) -> bool:
    """
    Vérifie les credentials de l'authentification HTTP Basic
    
    Args:
        auth_header: L'header Authorization complet
        
    Returns:
        bool: True si les credentials sont valides
    """
    try:
        if not auth_header or not auth_header.startswith("Basic "):
            return False
            
        # Extraire et décoder les credentials
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)
        
        # Comparaison sécurisée pour éviter les attaques de timing
        correct_username = secrets.compare_digest(username, ADMIN_USERNAME)
        correct_password = secrets.compare_digest(password, ADMIN_PASSWORD)
        
        return correct_username and correct_password
        
    except Exception:
        return False