import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "runs/detect/train/weights/best.pt"
VIDEO = "videos/tracking_demo.mp4"
OUTPUT = "reports/obstacle_reaction_demo.mp4"

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

frame_count = 0
alerts = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            imgsz=416, conf=0.25, verbose=False
        )[0]

        output = result.plot()
        left, right = int(w * 0.35), int(w * 0.65)
        cv2.rectangle(output, (left, 0), (right, h), (255, 200, 0), 2)

        obstacle = False
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = (x1 + x2) // 2
                if left <= cx <= right:
                    obstacle = True
                    break

        if obstacle:
            alerts += 1
            message = "OBSTACLE ALERT - OBJECT IN FORWARD ZONE"
            color = (0, 0, 255)
        else:
            message = "FORWARD ZONE CLEAR"
            color = (0, 255, 0)

        cv2.rectangle(output, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(output, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        writer.write(output)
        frame_count += 1

finally:
    cap.release()
    writer.release()

print("Frames processed:", frame_count)
print("Frames with obstacle alerts:", alerts)
print("Saved:", OUTPUT)
print("Note: This is a visual simulation, not real vehicle control.")
