# Perception API

A production-ready REST API for real-time object detection using YOLOv8. Upload an image, get back bounding boxes, class labels, and confidence scores — or receive the image with annotations drawn on it.

Built with FastAPI + Ultralytics YOLOv8. Designed for easy integration into ADAS pipelines, robotics systems, and computer vision applications.

---

## Features

- `/detect` — JSON response with bounding boxes, labels, confidence scores, inference time
- `/detect/annotated` — returns the image with detections drawn
- `/health` — model status and device info
- `/models` — list available YOLOv8 model variants
- Configurable confidence threshold per request
- CPU and CUDA GPU support (auto-detected)
- Clean Pydantic schemas, fully typed

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/usha-jujjuru/perception-api.git
cd perception-api
pip install -r requirements.txt

# 2. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open interactive docs
# http://localhost:8000/docs
```

---

## API Usage

### Detect objects (JSON response)

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@sample_images/test_street.jpg"
```

**Response:**
```json
{
  "model": "yolov8n",
  "image_width": 1280,
  "image_height": 720,
  "inference_time_ms": 18.4,
  "detection_count": 3,
  "detections": [
    {
      "label": "car",
      "confidence": 0.91,
      "class_id": 2,
      "bounding_box": { "x1": 120.0, "y1": 200.0, "x2": 450.0, "y2": 380.0 }
    }
  ]
}
```

### Get annotated image

```bash
curl -X POST "http://localhost:8000/detect/annotated" \
  -F "file=@your_image.jpg" \
  --output result.jpg
```

### Adjust confidence threshold

```bash
curl -X POST "http://localhost:8000/detect?conf=0.5" \
  -F "file=@your_image.jpg"
```

---

## Model Variants

| Model | Speed | Accuracy | Use case |
|-------|-------|----------|----------|
| yolov8n | Fastest | Lowest | Edge/embedded |
| yolov8s | Fast | Medium | Balanced |
| yolov8m | Medium | Good | General |
| yolov8l | Slow | High | High accuracy |
| yolov8x | Slowest | Highest | Offline batch |

Change the model in `app/main.py`:
```python
detector = ObjectDetector(model_name="yolov8s")
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
perception-api/
├── app/
│   ├── main.py        # FastAPI routes
│   ├── detector.py    # YOLOv8 inference wrapper
│   └── schemas.py     # Pydantic request/response models
├── tests/
│   └── test_api.py
├── sample_images/
│   ├── test_street.jpg              # Sample urban street scene
│   └── test_street_detections.json  # Sample detection output
├── requirements.txt
└── README.md
```

---

## Extending

- **Swap model**: change `model_name` in `ObjectDetector` to any Ultralytics-supported model including YOLOv10, YOLOv9, custom `.pt` weights
- **Custom weights**: `ObjectDetector(model_name="path/to/custom.pt")`
- **Add video streaming**: wire `/detect` into a WebSocket endpoint with OpenCV frame capture
- **Docker**: `docker build -t perception-api . && docker run -p 8000:8000 perception-api`

---

## Author

**Usha Rani Jujjuru**  
M.Sc. Automotive Software Engineering — TU Chemnitz  
Perception Engineer | Computer Vision | ADAS | Autonomous Driving  
[LinkedIn](https://linkedin.com/in/usha-rani-jujjuru) · [GitHub](https://github.com/usha-jujjuru)
