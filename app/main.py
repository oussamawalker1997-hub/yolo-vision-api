from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from app.inference import ObjectDetector
from app.schemas import InferenceResponse
from app.core.config import settings
from app.core.logger import logger

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage du service - Initialisation des modèles ML")
    ml_models["yolo"] = ObjectDetector()
    yield
    logger.info("Arrêt du service - Libération de la mémoire GPU/CPU")
    ml_models.clear()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Activation du monitoring Prometheus
Instrumentator().instrument(app).expose(app)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "operational", "model": settings.MODEL_PATH}

@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejet d'un fichier non-image: {file.content_type}")
        raise HTTPException(status_code=400, detail="Seules les images sont acceptées.")
    
    try:
        image_bytes = await file.read()
        results = ml_models["yolo"].predict(image_bytes)
        return InferenceResponse(
            filename=file.filename,
            objects_detected=len(results),
            detections=results
        )
    except Exception as e:
        logger.error(f"Erreur d'inférence: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du modèle ML.")
