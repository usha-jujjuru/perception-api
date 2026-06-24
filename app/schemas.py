from pydantic import BaseModel, Field, ConfigDict
from typing import List


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    class_id: int


class DetectionResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    inference_time_ms: float
    detections: List[Detection]
    detection_count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    device: str


class ModelInfo(BaseModel):
    name: str
    description: str


class ModelsResponse(BaseModel):
    available_models: List[ModelInfo]
    current_model: str
