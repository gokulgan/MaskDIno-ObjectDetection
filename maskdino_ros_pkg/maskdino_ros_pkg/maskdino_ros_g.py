#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy

# Add MaskDINO path
sys.path.append('/root/ros2_ws/src/maskdino_ros_pkg/maskdino_ros_pkg/maskdino/MaskDINO')
try:
    from detect import MaskDinoDetector
except ImportError:
    print("Warning: Could not import MaskDinoDetector. Using dummy detector.")

class SimpleMaskDINO2D(Node):
    def __init__(self):
        super().__init__('simple_maskdino_2d')
        
        # Parameters
        self.declare_parameters(namespace='', parameters=[
            ('weights_path', ''),
            ('config_path', ''),
            ('labels_path', ''),
            ('image_topic', '/femto_mega/color/image_raw'),
            ('depth_topic', '/femto_mega/depth/image_raw'),  # NEW: Depth image topic
            ('camera_info_topic', '/femto_mega/depth/camera_info'),  # NEW: Camera info topic
            ('output_topic', '/maskdino/detections_2d'),
            ('confidence_threshold', 0.3),
            ('visualize', True),
            ('target_frame', 'base_link'),  # NEW: Target frame for pose transformation
            ('enable_pose_estimation', True),  # NEW: Enable/disable pose estimation
        ])
        
        # Get parameters
        weights = self.get_parameter('weights_path').value
        config = self.get_parameter('config_path').value
        labels = self.get_parameter('labels_path').value
        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value
        conf_thresh = self.get_parameter('confidence_threshold').value
        visualize = self.get_parameter('visualize').value
        self.target_frame = self.get_parameter('target_frame').value
        self.enable_pose = self.get_parameter('enable_pose_estimation').value
        
        # Load class labels (11 classes)
        self.class_names = self.load_labels(labels)
        self.get_logger().info(f"Loaded {len(self.class_names)} classes: {self.class_names}")
        
        # Initialize MaskDINO detector
        try:
            self.detector = MaskDinoDetector(
                conf_thresh=conf_thresh,
                model_path=weights,
                config_path=config,
                class_names=self.class_names
            )
            self.get_logger().info("MaskDINO detector initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize MaskDINO: {e}")
            self.detector = None
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # NEW: 3D Pose Estimation Data
        self.latest_depth = None
        self.camera_info = None
        self.depth_frame_id = None
        self.color_width = None
        self.color_height = None
        self.depth_width = None
        self.depth_height = None
        
        # NEW: TF2 for coordinate transformations
        if self.enable_pose:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # QoS Profile for camera data
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile
        )
        
        # NEW: Depth image subscriber
        if self.enable_pose and depth_topic:
            self.depth_sub = self.create_subscription(
                Image, depth_topic, self.depth_callback, qos_profile
            )
            self.get_logger().info(f"   Depth topic: {depth_topic}")
        else:
            self.get_logger().warn("Pose estimation disabled - no depth topic provided")
        
        # NEW: Camera info subscriber
        if self.enable_pose and camera_info_topic:
            self.camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic, self.camera_info_callback, 10
            )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, output_topic, 10
        )
        
        # NEW: Pose publisher for 3D poses
        if self.enable_pose:
            self.pose_pub = self.create_publisher(
                PoseStamped, f"{output_topic}/poses", 10
            )
        
        if visualize:
            self.viz_pub = self.create_publisher(
                Image, f"{output_topic}/visualization", 10
            )
        else:
            self.viz_pub = None
        
        self.get_logger().info(f"✅ Simple MaskDINO 2D Node Ready")
        self.get_logger().info(f"   Subscribing to: {image_topic}")
        self.get_logger().info(f"   Publishing to: {output_topic}")
        if self.enable_pose:
            self.get_logger().info(f"   3D Pose estimation: ENABLED (target frame: {self.target_frame})")
    
    def load_labels(self, labels_path):
        """Load class labels from file"""
        default_labels = [
            "gun", "pringles_L", "pringles_S", "peach", "pear",
            "apple", "strawberry", "soccer_ball", "cola", "mustard", "cup"
        ]
        
        if not labels_path:
            return default_labels
        
        try:
            import json
            with open(labels_path, 'r') as f:
                data = json.load(f)
                if 'categories' in data:
                    return [cat['name'] for cat in data['categories']]
                else:
                    return data.get('labels', default_labels)
        except Exception as e:
            self.get_logger().warn(f"Could not load labels from {labels_path}: {e}")
            return default_labels
    
    # NEW: Depth callback
    def depth_callback(self, msg):
        """Callback for depth image"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_frame_id = msg.header.frame_id
            if self.depth_width is None:
                self.depth_height, self.depth_width = self.latest_depth.shape[:2]
                self.get_logger().info(f"Depth image size: {self.depth_width}x{self.depth_height}")
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
    
    # NEW: Camera info callback
    def camera_info_callback(self, msg):
        """Callback for camera info"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(f"Camera info received - fx: {msg.k[0]:.2f}, fy: {msg.k[4]:.2f}")
    
    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store image dimensions
            if self.color_width is None:
                self.color_height, self.color_width = cv_image.shape[:2]
                self.get_logger().info(f"Color image size: {self.color_width}x{self.color_height}")
            
            # Run detection
            if self.detector:
                instances, viz_image = self.detector.detect(cv_image)
                
                # Convert to ROS Detection2DArray (with optional 3D poses)
                detection_msg = self.create_detection_msg(msg, instances)
                
                # Publish detections
                self.detection_pub.publish(detection_msg)
                
                # NEW: Add 3D coordinates to visualization if enabled
                if self.viz_pub and viz_image is not None:
                    if self.enable_pose and len(instances.pred_boxes) > 0:
                        viz_image = self.add_3d_info_to_viz(viz_image, instances)
                    
                    viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
                    viz_msg.header = msg.header
                    self.viz_pub.publish(viz_msg)
                
                # Log info
                if len(instances.pred_boxes) > 0:
                    self.get_logger().info(
                        f"Detected {len(instances.pred_boxes)} objects"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    # NEW: Add 3D coordinates to visualization
    def add_3d_info_to_viz(self, viz_image, instances):
        """Add 3D pose information to visualization image"""
        try:
            for i in range(len(instances.pred_boxes)):
                # Get bounding box
                box = instances.pred_boxes[i].tensor[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Estimate 3D pose
                pose = self.estimate_pose(cx, cy)
                
                # Draw 3D coordinates if valid
                if pose.position.z > 0:
                    # Draw center dot
                    cv2.circle(viz_image, (cx, cy), 5, (0, 255, 255), -1)
                    
                    # Draw 3D coordinates text
                    pose_text = f"X:{pose.position.x:.2f} Y:{pose.position.y:.2f} Z:{pose.position.z:.2f}m"
                    cv2.putText(viz_image, pose_text, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Get class name if available
                    if hasattr(instances, 'pred_classes'):
                        class_id = int(instances.pred_classes[i].cpu().numpy())
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        self.get_logger().info(f"Found {class_name}: {pose_text}")
        except Exception as e:
            self.get_logger().error(f"Error adding 3D info to viz: {e}")
        
        return viz_image
    
    # NEW: Estimate 3D pose from 2D pixel coordinates
    def estimate_pose(self, cx, cy):
        """Converts 2D pixel (cx, cy) -> 3D Pose in Robot Frame"""
        pose = Pose()
        pose.orientation.w = 1.0  # Default orientation
        
        # Safety checks
        if not self.enable_pose:
            return pose
        
        if self.latest_depth is None or self.camera_info is None:
            return pose  # Returns empty pose (0,0,0)

        # Scaling (If RGB and Depth resolutions are different)
        if self.color_width != self.depth_width:
            scale_x = self.depth_width / self.color_width
            scale_y = self.depth_height / self.color_height
            dcx = int(cx * scale_x)
            dcy = int(cy * scale_y)
        else:
            dcx, dcy = cx, cy

        # Boundary check
        if not (0 <= dcx < self.depth_width and 0 <= dcy < self.depth_height):
            return pose

        # Get Depth (Distance Z)
        # Sample a 3x3 region to avoid getting a 0 value from a noisy pixel
        depth_region = self.latest_depth[max(0, dcy-1):min(self.depth_height, dcy+2), 
                                        max(0, dcx-1):min(self.depth_width, dcx+2)]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) == 0:
            return pose
            
        z_metric = float(np.median(valid_depths)) / 1000.0  # mm to meters

        # Deproject to 3D (Camera Frame)
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        x_cam = (dcx - ppx) * z_metric / fx
        y_cam = (dcy - ppy) * z_metric / fy
        z_cam = z_metric

        pose.position.x = x_cam
        pose.position.y = y_cam
        pose.position.z = z_cam

        # Transform to Robot Frame (base_link or target_frame)
        if self.depth_frame_id and self.target_frame:
            pose = self.transform_pose(pose, self.depth_frame_id, self.target_frame)
            
        return pose
    
    # NEW: Transform pose using TF2
    def transform_pose(self, input_pose, from_frame, to_frame):
        """Transforms pose using TF2"""
        try:
            pose_stamped = PoseStamped()
            pose_stamped.pose = input_pose
            pose_stamped.header.frame_id = from_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            # Wait for transform (1 second timeout)
            if not self.tf_buffer.can_transform(to_frame, from_frame, rclpy.time.Time()):
                self.get_logger().warn(f"Cannot transform from {from_frame} to {to_frame}")
                return input_pose

            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, timeout=Duration(seconds=1.0))
            return output_pose_stamped.pose
        except TransformException as e:
            self.get_logger().warn(f"TF Error: {e}")
            return input_pose
    
    def create_detection_msg(self, img_msg, instances):
        """Convert MaskDINO instances to ROS Detection2DArray"""
        detection_array = Detection2DArray()
        detection_array.header = img_msg.header
        
        if not hasattr(instances, 'pred_boxes'):
            return detection_array
        
        for i in range(len(instances.pred_boxes)):
            # Get bounding box
            box = instances.pred_boxes[i].tensor[0].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Create Detection2D
            detection = Detection2D()
            detection.header = img_msg.header
            
            # Set bounding box
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Set results (class and confidence)
            if hasattr(instances, 'scores'):
                score = float(instances.scores[i].cpu().numpy())
                class_id = int(instances.pred_classes[i].cpu().numpy())
                
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id)
                hypothesis.hypothesis.score = score
                
                # NEW: Add 3D pose to hypothesis if available
                if self.enable_pose:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    pose_3d = self.estimate_pose(cx, cy)
                    hypothesis.pose.pose = pose_3d
                    
                    # Publish individual pose
                    if self.pose_pub and pose_3d.position.z > 0:
                        pose_stamped = PoseStamped()
                        pose_stamped.header = img_msg.header
                        pose_stamped.pose = pose_3d
                        self.pose_pub.publish(pose_stamped)
                
                detection.results.append(hypothesis)
            
            # Add mask if available
            if hasattr(instances, 'pred_masks'):
                mask = instances.pred_masks[i].cpu().numpy()
                # You could encode mask as Image message if needed
            
            detection_array.detections.append(detection)
        
        return detection_array

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMaskDINO2D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
