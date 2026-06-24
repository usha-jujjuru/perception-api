import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_test_image(width: int = 640, height: int = 480) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["device"] in ("cpu", "cuda")


def test_list_models():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["available_models"]) > 0
    assert data["current_model"] == "yolov8n"


def test_detect_returns_valid_schema():
    img_bytes = make_test_image()
    r = client.post("/detect", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "detections" in data
    assert "inference_time_ms" in data
    assert data["image_width"] == 640
    assert data["image_height"] == 480


def test_detect_annotated_returns_image():
    img_bytes = make_test_image()
    r = client.post("/detect/annotated", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_detect_invalid_file():
    r = client.post("/detect", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_detect_confidence_threshold():
    img_bytes = make_test_image()
    r = client.post("/detect?conf=0.9", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
