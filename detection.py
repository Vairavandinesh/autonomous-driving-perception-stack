from ultralytics import YOLO
from pathlib import Path
import csv

model = YOLO("yolo11n.pt")
image_path = "bus.jpg"

results = model.predict(
    source=image_path,
    imgsz=416,
    conf=0.4,
    save=True,
    project="runs/detect",
    name="baseline",
    exist_ok=True
)

output_file = Path("reports") / "baseline_detections.csv"
output_file.parent.mkdir(exist_ok=True)

with output_file.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            writer.writerow([
                class_id,
                model.names[class_id],
                round(float(box.conf.item()), 4),
                *[round(float(v), 2) for v in box.xyxy[0].tolist()]
            ])

print(f"Detected objects saved to {output_file}")