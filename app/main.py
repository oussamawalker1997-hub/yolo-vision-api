from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from app.inference import ObjectDetector
from app.schemas import InferenceResponse

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["yolo"] = ObjectDetector("yolov8n.pt")
    yield
    ml_models.clear()

app = FastAPI(
    title="MLOps Vision API",
    description="API de production pour la détection d'objets (YOLOv8)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "operational", "model": "YOLOv8 Nano"}

@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier non supporté.")
    try:
        image_bytes = await file.read()
        results = ml_models["yolo"].predict(image_bytes)
        return InferenceResponse(filename=file.filename, objects_detected=len(results), detections=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
