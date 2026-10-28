import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

image_path = "datasets/lane/road.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

h, w = image.shape[:2]
model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

print("Loading pretrained segmentation model...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
model.eval()

inputs = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                   return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

segmentation = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[(h, w)]
)[0].cpu().numpy()

labels = model.config.id2label
road_ids = [int(k) for k, v in labels.items() if v.lower() == "road"]

if not road_ids:
    raise RuntimeError("Road class not found in model labels")

road_mask = np.isin(segmentation, road_ids)
overlay = image.copy()
overlay[road_mask] = (
    0.5 * overlay[road_mask] + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
roi_mask = np.zeros_like(edges)
polygon = np.array([[
    (0, h), (int(w * .45), int(h * .6)),
    (int(w * .55), int(h * .6)), (w, h)
]], np.int32)
cv2.fillPoly(roi_mask, polygon, 255)

lines = cv2.HoughLinesP(
    cv2.bitwise_and(edges, roi_mask), 1, np.pi / 180,
    25, minLineLength=20, maxLineGap=150
)

line_count = 0
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4].astype(int)
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
        line_count += 1

cv2.putText(overlay, "Road segmentation: blue | OpenCV lines: red",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imwrite("reports/segmentation_comparison.jpg", overlay)
print("Road pixels:", int(road_mask.sum()))
print("OpenCV line segments:", line_count)
print("Saved: reports/segmentation_comparison.jpg")
