import cv2
import torch
import numpy as np

image = cv2.imread("bus.jpg")
if image is None:
    raise FileNotFoundError("bus.jpg not found")

model_type = "MiDaS_small"
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

input_batch = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
with torch.no_grad():
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()
depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
cv2.imwrite("reports/midas_depth_result.jpg", colored_depth)
print("Saved: reports/midas_depth_result.jpg")
print("Note: MiDaS estimates relative depth, not metric distance in meters.")
