import argparse
import queue
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt")
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        raise SystemExit(
            "CARLA Python API not found. Install CARLA and configure its Python API."
        )

    model = YOLO(args.model)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicles = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No vehicle spawn points available.")

    vehicle = world.try_spawn_actor(vehicles[0], spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Could not spawn vehicle. Try another CARLA map.")

    camera = None
    frames = queue.Queue(maxsize=2)

    try:
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        camera = world.spawn_actor(
            camera_bp, camera_transform, attach_to=vehicle
        )

        def receive_image(image):
            image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = image_data.reshape((image.height, image.width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frames.full():
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
            frames.put(frame)

        camera.listen(receive_image)
        print("CARLA camera running. Press Q to exit.")

        while True:
            world.wait_for_tick()
            try:
                frame = frames.get(timeout=2)
            except queue.Empty:
                continue

            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                imgsz=416, conf=0.25, verbose=False
            )[0]

            annotated = result.plot()
            cv2.imshow("CARLA Perception", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
