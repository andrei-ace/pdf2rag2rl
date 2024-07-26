import os
import requests
from ultralytics import YOLO

def download_model_weights(url, save_path):
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Weights already exist at {save_path}")

weights_url = "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt"
weights_path = "models/yolov10x_best.pt"

download_model_weights(weights_url, weights_path)
# Load the YOLOv10 model
model = YOLO(weights_path)

def detect_layout_elements(image_pil, model=model):
    results = model(image_pil, conf=0.2, iou=0.8)
    boxes = []
    for r in results:
        for bbox in r.boxes.xyxy.tolist():
            boxes.append(bbox)
    return boxes