import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEO = "videos/tracking_demo.mp4"
YOLO_MODEL = "runs/detect/train/weights/best.pt"
OUTPUT = "reports/integrated_perception.mp4"

# Load trained detector and MiDaS depth model
detector = YOLO(YOLO_MODEL)
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

frame_count = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO + ByteTrack
        result = detector.track(
            frame, persist=True, tracker="bytetrack.yaml",
            imgsz=416, conf=0.25, verbose=False
        )[0]
        annotated = result.plot()

        # Classical lane-line processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height), (int(width * .45), int(height * .6)),
            (int(width * .55), int(height * .6)), (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        lines = cv2.HoughLinesP(
            cv2.bitwise_and(edges, mask), 1, np.pi / 180,
            25, minLineLength=20, maxLineGap=150
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4].astype(int)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # MiDaS relative-depth visualization
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transform(rgb)
        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth = prediction.cpu().numpy()
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        depth_color = cv2.resize(depth_color, (width // 3, height // 3))
        annotated[0:height // 3, 0:width // 3] = depth_color

        writer.write(annotated)
        frame_count += 1
        if frame_count % 25 == 0:
            print(f"Processed {frame_count} frames")

finally:
    cap.release()
    writer.release()

print(f"Done. Processed {frame_count} frames.")
print(f"Saved video: {OUTPUT}")
print("Depth is relative only; it is not distance in meters.")
