import argparse
import time
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "NVIDIA CUDA GPU unavailable. TensorRT execution cannot run "
            "on this AMD CPU-only system."
        )

    model = YOLO(args.model)
    engine_path = model.export(
        format="engine", imgsz=args.imgsz, half=True, device=0
    )
    engine = YOLO(engine_path)

    for _ in range(10):
        engine.predict("bus.jpg", imgsz=args.imgsz, device=0, verbose=False)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(args.iterations):
        engine.predict("bus.jpg", imgsz=args.imgsz, device=0, verbose=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency = elapsed * 1000 / args.iterations
    print(f"Engine: {engine_path}")
    print(f"Average latency: {latency:.2f} ms")
    print(f"Approximate FPS: {1000 / latency:.2f}")

if __name__ == "__main__":
    main()
