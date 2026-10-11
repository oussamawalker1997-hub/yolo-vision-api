from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class DetectionResult(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox

class InferenceResponse(BaseModel):
    filename: str
    objects_detected: int
    detections: List[DetectionResult]
