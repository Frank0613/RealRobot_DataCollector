"""
Subscribes to a real robot's joint_state + RGB + depth topics and exposes
the latest *time-synchronized* triple to the main simulation loop.

Synchronization is done by message_filters.ApproximateTimeSynchronizer,
so every cached snapshot carries three messages whose header.stamps lie
within `slop` seconds of each other.

rclpy spins in a background thread to avoid blocking Isaac Sim's main loop.
"""
import threading
import numpy as np


class RealRobotSubscriber:
    def __init__(
        self,
        joint_topic: str,
        rgb_topic: str,
        depth_topic: str,
        rgb_compressed: bool = True,
        slop: float = 0.05,
        queue_size: int = 10,
        node_name: str = "real_robot_subscriber",
    ):
        # Lazy imports — these only resolve once the Isaac Sim ROS2 bridge
        # extension has been enabled and rclpy is on sys.path.
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        import message_filters
        from sensor_msgs.msg import JointState, Image, CompressedImage

        self._rclpy = rclpy
        self._rgb_compressed = rgb_compressed
        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)

        rgb_msg_type = CompressedImage if rgb_compressed else Image
        joint_sub = message_filters.Subscriber(self._node, JointState, joint_topic)
        rgb_sub   = message_filters.Subscriber(self._node, rgb_msg_type, rgb_topic)
        depth_sub = message_filters.Subscriber(self._node, Image, depth_topic)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [joint_sub, rgb_sub, depth_sub],
            queue_size,
            slop,
        )
        self._sync.registerCallback(self._on_synced)

        self._lock = threading.Lock()
        self._latest = None        # dict of synced arrays
        self._joint_names = None   # captured from first joint message
        self._first_logged = False

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="ros_subscriber"
        )
        self._thread.start()

        rgb_kind = "CompressedImage" if rgb_compressed else "Image"
        print("[RealRobotSubscriber] spinning, topics:")
        print(f"  joint = {joint_topic}")
        print(f"  rgb   = {rgb_topic}  ({rgb_kind})")
        print(f"  depth = {depth_topic}")
        print(f"  slop  = {slop}s, queue={queue_size}")

    # ---------- callback ----------

    def _on_synced(self, joint_msg, rgb_msg, depth_msg):
        joint_pos = np.asarray(joint_msg.position, dtype=np.float32)
        joint_vel = (
            np.asarray(joint_msg.velocity, dtype=np.float32)
            if len(joint_msg.velocity) == len(joint_msg.position)
            else np.zeros_like(joint_pos)
        )

        rgb   = (self._compressed_msg_to_array(rgb_msg)
                 if self._rgb_compressed
                 else self._image_msg_to_array(rgb_msg))
        depth = self._image_msg_to_array(depth_msg)

        stamp = rgb_msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        with self._lock:
            if self._joint_names is None and joint_msg.name:
                self._joint_names = list(joint_msg.name)
            self._latest = {
                "real_joint_pos": joint_pos,
                "real_joint_vel": joint_vel,
                "real_rgb": rgb,
                "real_depth": depth,
                "real_stamp": np.float64(t),
            }

        if not self._first_logged:
            self._first_logged = True
            print(
                f"[RealRobotSubscriber] first synced frame: "
                f"joints={joint_pos.shape}, rgb={rgb.shape}, depth={depth.shape}"
            )

    # ---------- helpers ----------

    @staticmethod
    def _image_msg_to_array(msg):
        """Decode sensor_msgs/Image into a numpy array. Always returns RGB
        for color images. Depth is left in its native dtype (uint16 mm or
        float32 m)."""
        h, w, enc = msg.height, msg.width, msg.encoding

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc == "rgba8" or enc == "bgra8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[..., :3]
            if enc == "bgra8":
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        if enc in ("16UC1", "mono16"):
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).copy()

        if enc == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()

        if enc == "mono8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w).copy()

        # Unknown — store raw bytes reshaped as best-effort
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1).copy()

    @staticmethod
    def _compressed_msg_to_array(msg):
        """Decode sensor_msgs/CompressedImage (JPEG/PNG) into an RGB uint8 array."""
        import cv2  # ships with Isaac Sim's Python
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"cv2.imdecode failed (format='{msg.format}', {len(msg.data)} bytes)"
            )
        return np.ascontiguousarray(bgr[..., ::-1])  # BGR → RGB

    # ---------- public API ----------

    def get_latest(self):
        """Latest time-synchronized triple as a dict, or None if we have
        not yet received any synced messages."""
        with self._lock:
            return self._latest

    def is_ready(self) -> bool:
        return self.get_latest() is not None

    @property
    def real_joint_names(self):
        return self._joint_names

    def shutdown(self):
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        # Don't call rclpy.shutdown() — other Isaac Sim components may share it.
