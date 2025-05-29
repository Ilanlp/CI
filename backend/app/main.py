import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import router
from app.api.routes2 import router2
from app.api.routes_competence import router_competence
from app.api.routes_contrat import router_contrat
from app.api.routes_lieu import router_lieu
from app.api.routes_metier import router_metier
from app.api.routes_romecode import router_romecode
from app.api.routes_seniorite import router_seniorite
from app.auth import verify_basic_auth
import time
import uvicorn

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    openapi_tags=[
        {"name": "Système", "description": "Opérations système"},
        {"name": "Références", "description": "Tables de références"},
        {"name": "Relationnels", "description": "Tables relationnels"},
        {"name": "Modèles", "description": "Accès au version du modèle"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Équipe de support Jobmarket",
        "email": "support@jobmarket.com",
        "url": "https://api.jobmarket/support",
    },
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A ajuster pour la production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration CORS (votre code existant)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware d'authentification
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Middleware d'authentification pour les méthodes PUT, POST, DELETE"""
    
    # Vérifier si la méthode nécessite une authentification
    if request.method in ["PUT", "POST", "DELETE", "PATCH"]:
        print(f"Authentification requise pour {request.method} {request.url.path}")  # Debug
        
        # Récupérer l'header d'autorisation
        auth_header = request.headers.get("authorization")
        print(f"Auth header reçu: {auth_header}")  # Debug
        
        # Vérifier les credentials
        if not verify_basic_auth(auth_header):
            print("Authentification échouée")  # Debug
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "message": "Authentification requise pour cette opération",
                    "data": None,
                },
                headers={"WWW-Authenticate": "Basic realm=\"API\""},
            )
        
        print("Authentification réussie")  # Debug
    
    # Si authentifié ou méthode non soumise à auth, continuer
    response = await call_next(request)
    return response

# Middleware pour le logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s"
    )

    return response


# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Exception non gérée: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Une erreur interne s'est produite",
            "data": None,
        },
    )


# Inclusion des routes
app.include_router(router, prefix="/api")
app.include_router(router2, prefix="/api2")
app.include_router(router_competence, prefix="/api")
app.include_router(router_contrat,prefix="/api")
app.include_router(router_lieu,prefix="/api")
app.include_router(router_metier,prefix="/api")
app.include_router(router_romecode,prefix="/api")
app.include_router(router_seniorite,prefix="/api")
# Route de santé
@app.get("/", tags=["Système"])
async def root():
    return {"status": "ok", "version": settings.API_VERSION}


# Route de santé
@app.get("/health_check", tags=["Système"])
async def health_check():
    return {"status": "ok", "version": settings.API_VERSION}


# Route de santé
@app.get("/health/modele/v1", tags=["Système"])
async def health_v1():
    return {"status": "ok", "version": settings.API_VERSION}


# Route de santé
@app.get("/health/modele/v2", tags=["Système"])
async def health_v2():
    return {"status": "ok", "version": settings.API_VERSION}

# Pour lancer l'application directement avec python
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
