import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter(
            "model_path", "runs/detect/train/weights/best.pt"
        )
        model_path = self.get_parameter("model_path").value

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.image_pub = self.create_publisher(
            Image, "/perception/annotated_image", 10
        )
        self.alert_pub = self.create_publisher(
            Bool, "/perception/obstacle_alert", 10
        )

        self.get_logger().info("ROS 2 perception node started")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            height, width = frame.shape[:2]

            result = self.model(
                frame, imgsz=416, conf=0.25, verbose=False
            )[0]

            obstacle = False

            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                center_x = (x1 + x2) // 2

                if width * 0.35 <= center_x <= width * 0.65:
                    obstacle = True

            alert = Bool()
            alert.data = obstacle
            self.alert_pub.publish(alert)

            annotated = result.plot()
            output = self.bridge.cv2_to_imgmsg(
                annotated, encoding="bgr8"
            )
            output.header = msg.header
            self.image_pub.publish(output)

        except Exception as error:
            self.get_logger().error(str(error))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
