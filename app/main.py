import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from app.detector import ObjectDetector, AVAILABLE_MODELS
from app.schemas import (
    DetectionResponse,
    HealthResponse,
    ModelsResponse,
    ModelInfo,
)

app = FastAPI(
    title="Perception API",
    description=(
        "Real-time object detection API powered by YOLOv8. "
        "Upload an image to get bounding boxes, labels, and confidence scores."
    ),
    version="1.0.0",
)

detector = ObjectDetector(model_name="yolov8n", conf_threshold=0.25)


def decode_image(file: UploadFile) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        return np.array(img)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=detector.is_loaded,
        device=detector.device,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Status"])
def list_models():
    return ModelsResponse(
        available_models=[
            ModelInfo(name=k, description=v) for k, v in AVAILABLE_MODELS.items()
        ],
        current_model=detector.model_name,
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="Image file (jpg, png, bmp)"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Detect objects in an uploaded image.
    Returns bounding boxes, class labels, confidence scores, and inference time.
    """
    detector.conf_threshold = conf
    image = decode_image(file)
    result = detector.detect(image)
    return DetectionResponse(**result)


@app.post("/detect/annotated", tags=["Detection"])
async def detect_annotated(
    file: UploadFile = File(..., description="Image file (jpg, png, bmp)"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Detect objects and return the image with bounding boxes drawn on it.
    """
    detector.conf_threshold = conf
    image = decode_image(file)
    annotated = detector.detect_and_annotate(image)
    annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR → RGB
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
