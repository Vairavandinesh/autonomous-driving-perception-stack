import time
import cv2
import onnxruntime as ort
import numpy as np

model_path = "runs/detect/train/weights/best.onnx"
image_path = "bus.jpg"

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
image = cv2.imread(image_path)
image = cv2.resize(image, (416, 416))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
input_tensor = np.transpose(image, (2, 0, 1))[None, ...]

input_name = session.get_inputs()[0].name
for _ in range(3):
    session.run(None, {input_name: input_tensor})

times = []
for _ in range(10):
    start = time.perf_counter()
    session.run(None, {input_name: input_tensor})
    times.append((time.perf_counter() - start) * 1000)

print(f"ONNX CPU average latency: {np.mean(times):.2f} ms")
print(f"Approximate FPS: {1000 / np.mean(times):.2f}")
