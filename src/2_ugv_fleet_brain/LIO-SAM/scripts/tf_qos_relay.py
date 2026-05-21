#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_msgs.msg import TFMessage


class TfQosRelay(Node):
    def __init__(self):
        super().__init__("tf_qos_relay")

        self.latest_tf = None

        transient_qos = QoSProfile(depth=10)
        transient_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        volatile_qos = QoSProfile(depth=10)
        volatile_qos.durability = DurabilityPolicy.VOLATILE

        self.sub_tf_static = self.create_subscription(
            TFMessage,
            "/tf_static",
            self.tf_static_callback,
            transient_qos,
        )

        self.pub_tf_static_lio = self.create_publisher(
            TFMessage,
            "/tf_static_lio_sam",
            volatile_qos,
        )

        self.republish_timer = self.create_timer(1.0, self.republish_callback)

        self.get_logger().info(
            "tf_qos_relay started: /tf_static (transient_local) -> /tf_static_lio_sam (volatile)"
        )

    def tf_static_callback(self, msg: TFMessage):
        self.latest_tf = msg

    def republish_callback(self):
        if self.latest_tf is not None:
            self.pub_tf_static_lio.publish(self.latest_tf)


def main(args=None):
    rclpy.init(args=args)
    node = TfQosRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
