import io
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

class ObjectDetector:
    def __init__(self, model_version: str = "yolov8n.pt"):
        self.model = YOLO(model_version)

    def predict(self, image_bytes: bytes) -> List[Dict]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Impossible de décoder l'image fournie.")

        results = self.model.predict(img, conf=0.25, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()
                c = box.cls
                conf = box.conf[0].item()
                class_name = self.model.names[int(c)]
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}
                })
        return detections
