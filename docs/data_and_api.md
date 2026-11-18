# Data and API Documentation

## Dataset
KITTI object detection dataset.
- Total annotations analyzed: 32,442
- Classes: 8
- Most frequent class: Car (23,026 annotations)

Dataset files are excluded from this repository and must be downloaded separately.

## Pipeline Components

| Component | Input | Output |
|---|---|---|
| YOLO11n | Image/video frame | Object classes, confidence, bounding boxes |
| ByteTrack | Object detections | Tracking IDs |
| OpenCV | Road image | Lane-line overlay |
| SegFormer-B0 | Road image | Road-surface segmentation |
| MiDaS Small | Image | Relative-depth map |
| ONNX Runtime | Image tensor | Model inference output |
| Obstacle alert | Video detections | Forward-zone warning |

## Main Scripts
- `dataset_exploration.py`: Dataset exploration.
- `detection.py`: Object detection.
- `lane_detection.py`: OpenCV lane detection.
- `segmentation_comparison.py`: Road-surface segmentation comparison.
- `depth_estimation.py`: MiDaS depth estimation.
- `integrated_perception.py`: Integrated perception pipeline.
- `obstacle_reaction.py`: Visual obstacle alert simulation.
- `onnx_benchmark.py`: ONNX CPU benchmark.

## Performance
YOLO11n validation after one training epoch:
- Precision: 42.5%
- Recall: 25.1%
- mAP50: 25.6%
- mAP50-95: 13.0%

ONNX Runtime CPU benchmark:
- Average latency: 33.85 ms
- Approximate inference speed: 29.54 FPS

## Limitations
- MiDaS produces relative depth, not metric distance.
- SegFormer identifies road surfaces, not lane markings.
- Obstacle alerts are visual demonstrations, not vehicle control.
- CARLA, ROS 2, and TensorRT integration are not implemented.
- No real-vehicle testing has been performed.
