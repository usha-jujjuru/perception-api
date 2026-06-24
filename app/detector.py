import time
import numpy as np
from ultralytics import YOLO
import torch

AVAILABLE_MODELS = {
    "yolov8n": "YOLOv8 Nano — fastest, lowest accuracy",
    "yolov8s": "YOLOv8 Small — balanced speed/accuracy",
    "yolov8m": "YOLOv8 Medium — higher accuracy",
    "yolov8l": "YOLOv8 Large — high accuracy",
    "yolov8x": "YOLOv8 XLarge — highest accuracy, slowest",
}


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n", conf_threshold: float = 0.25):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(self.device)

    def detect(self, image: np.ndarray) -> dict:
        start = time.perf_counter()
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": results.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class_id": int(box.cls),
            })

        h, w = image.shape[:2]
        return {
            "model": self.model_name,
            "image_width": w,
            "image_height": h,
            "inference_time_ms": round(elapsed_ms, 2),
            "detections": detections,
            "detection_count": len(detections),
        }

    def detect_and_annotate(self, image: np.ndarray) -> np.ndarray:
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        return results.plot()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
