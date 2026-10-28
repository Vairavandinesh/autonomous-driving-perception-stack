import cv2
import numpy as np
from pathlib import Path

Path("reports").mkdir(exist_ok=True)
image = cv2.imread("bus.jpg")

if image is None:
    raise FileNotFoundError("bus.jpg not found")

height, width = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

mask = np.zeros_like(edges)
polygon = np.array([[
    (0, height), (int(width * 0.4), int(height * 0.55)),
    (int(width * 0.6), int(height * 0.55)), (width, height)
]], dtype=np.int32)
cv2.fillPoly(mask, polygon, 255)
masked_edges = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 40,
                        minLineLength=30, maxLineGap=100)
result = image.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = np.array(line).reshape(-1, 4)[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imwrite("reports/lane_detection_result.jpg", result)
print("Saved: reports/lane_detection_result.jpg")

