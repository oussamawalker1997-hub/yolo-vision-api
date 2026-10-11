import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from app.core.config import settings
from app.core.logger import logger

class ObjectDetector:
    def __init__(self):
        logger.info(f"Chargement du modèle YOLO depuis: {settings.MODEL_PATH}")
        self.model = YOLO(settings.MODEL_PATH)
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD

    def predict(self, image_bytes: bytes) -> List[Dict]:
        logger.info("Décodage de l'image pour inférence...")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Échec du décodage de l'image.")
            raise ValueError("Format d'image invalide ou corrompu.")

        results = self.model.predict(img, conf=self.conf_threshold, verbose=settings.DEBUG_MODE)
        detections = []
        
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                c = int(box.cls)
                detections.append({
                    "class_name": self.model.names[c],
                    "confidence": box.conf[0].item(),
                    "box": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}
                })
        
        logger.info(f"Inférence terminée: {len(detections)} objets détectés.")
        return detections
