from ultralytics import YOLO
import cv2
from pathlib import Path

# Load lightweight YOLOv8n
model = YOLO("yolov8n.pt")

# Test on one frame from Butterfly stroke
frames = list(Path("data/V2/videos/frames").glob("*/*.jpg"))
print(f"Total frames available: {len(frames)}")

if frames:
    sample_frame = str(frames[0])
    results = model(sample_frame, verbose=False)
    for r in results:
        boxes = r.boxes
        persons = [b for b in boxes if int(b.cls[0]) == 0]
        print(f"Sample: {sample_frame}")
        print(f"Detected {len(persons)} person(s) with confidences: {[round(float(p.conf[0]), 2) for p in persons]}")
