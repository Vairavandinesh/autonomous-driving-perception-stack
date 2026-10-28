# AIML-04: Autonomous Driving Perception Stack

## Components
- YOLO object detection fine-tuned on KITTI
- ByteTrack object tracking
- OpenCV lane-edge processing
- MiDaS relative-depth estimation
- ONNX CPU inference benchmark

## Results
- KITTI training labels: 32,442 objects across 8 classes
- One-epoch validation: Precision 42.5%, Recall 25.1%, mAP50 25.6%, mAP50-95 13.0%
- ONNX CPU inference: 33.85 ms average latency, approximately 29.54 FPS

## Reports
See the `reports/` folder for CSV reports and generated images.
The tracking video is in `runs/detect/runs/bytetrack_demo/`.

## Setup
Python 3.11. Install dependencies:
pip install ultralytics opencv-python numpy pandas matplotlib timm onnxruntime

## Limitations
Depth is relative, not measured in meters. Lane detection needs road-specific evaluation. Results are preliminary. CARLA/ROS 2 and TensorRT integration are not yet completed. This prototype is not for real-vehicle deployment.
