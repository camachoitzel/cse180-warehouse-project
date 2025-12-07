#!/usr/bin/env python3
"""
Map-Aware Warehouse Human Detection Controller (Strict Filtering Version)
========================================================================

FIX: Eliminates "X-Ray Vision" (detecting people through/as shelves) by:
1. Aggressively filtering static map obstacles (Dense Grid Check).
2. Verifying the detected object distance matches the expected distance.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import math
import numpy as np
import time

class WarehouseControllerStrict(Node):
    def __init__(self):
        super().__init__('warehouse_controller_strict')
        
        # --- CONFIGURATION ---
        self.person1_expected = {'x': 1.00, 'y': -1.00, 'name': 'Person 1'}
        self.person2_expected = {'x': -12.00, 'y': 15.00, 'name': 'Person 2'}
        self.robot_start = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}
        
        # TUNING PARAMETERS
        self.config = {
            # Static Object Filtering (The Fix for the Shelf Issue)
            'static_buffer_radius': 0.25,      # Meters to ignore around map obstacles (Increased)
            'map_occupancy_threshold': 50,     # Value 0-100 to consider "occupied"
            
            # Human Shape Logic
            'human_width_min': 0.10,
            'human_width_max': 1.00,           # Tighter max width to exclude long walls
            'min_cluster_points': 4,
            'cluster_gap_threshold': 0.3,      # Meters max gap between points
            
            # Confidence
            'location_match_tolerance': 1.5,   # How close (m) detected object must be to expected
        }
        
        # State
        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.map_received = False
        self.mission_active = False

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        
        # Nav2
        self.navigator = BasicNavigator()

        # Start Mission Timer
        self.create_timer(1.0, self.mission_loop)
        self.get_logger().info('Strict Controller Initialized. Waiting for Map...')

    # ================= Callbacks =================
    def map_cb(self, msg):
        self.map_info = msg.info
        # Reshape for easy (row, col) access
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        if not self.map_received:
            self.get_logger().info(f'Map Received: {msg.info.width}x{msg.info.height}')
            self.map_received = True

    def scan_cb(self, msg):
        self.latest_scan = msg

    # ================= Helpers =================
    def get_robot_pose(self):
        """Get precise robot pose from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            # Quaternion to Yaw
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def world_to_map(self, wx, wy):
        """Convert World (m) to Map (px)."""
        if not self.map_info: return None
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        res = self.map_info.resolution
        mx = int((wx - ox) / res)
        my = int((wy - oy) / res)
        if 0 <= mx < self.map_info.width and 0 <= my < self.map_info.height:
            return mx, my
        return None

    def is_static_obstacle(self, wx, wy):
        """
        STRICT CHECK: Returns True if (wx, wy) is part of the static map (e.g. a shelf).
        Checks a DENSE GRID of pixels around the point.
        """
        if self.map_data is None: return False
        
        center_px = self.world_to_map(wx, wy)
        if not center_px: return False
        
        cx, cy = center_px
        
        # Convert buffer radius (meters) to pixels
        radius_px = int(self.config['static_buffer_radius'] / self.map_info.resolution)
        
        # Check square window around the point
        # If ANY pixel in this window is occupied, we assume the laser hit the static object.
        for r in range(cy - radius_px, cy + radius_px + 1):
            for c in range(cx - radius_px, cx + radius_px + 1):
                if 0 <= r < self.map_info.height and 0 <= c < self.map_info.width:
                    val = self.map_data[r, c]
                    # -1 is unknown, 100 is occupied. 
                    # We treat > 50 as obstacle.
                    if val > self.config['map_occupancy_threshold']:
                        return True
        return False

    def scan_for_human(self, expected_pose):
        """
        Process scan to find a human near the expected pose.
        Returns: (found_bool, detected_location_dict)
        """
        if not self.latest_scan or not self.map_received:
            return False, None

        robot_pose = self.get_robot_pose()
        if not robot_pose: return False, None
        rx, ry, ryaw = robot_pose

        # 1. Extract Dynamic Points (Filter out Shelves/Walls)
        dynamic_points = []
        ranges = self.latest_scan.ranges
        angle_min = self.latest_scan.angle_min
        inc = self.latest_scan.angle_increment

        for i, r in enumerate(ranges):
            # Range Filter
            if r < 0.2 or r > 5.0 or not math.isfinite(r): continue

            # Calc World Position
            angle = ryaw + angle_min + (i * inc)
            wx = rx + r * math.cos(angle)
            wy = ry + r * math.sin(angle)

            # --- THE FIX: STRICT STATIC FILTERING ---
            if not self.is_static_obstacle(wx, wy):
                dynamic_points.append({'x': wx, 'y': wy, 'idx': i})

        if len(dynamic_points) < self.config['min_cluster_points']:
            return False, None

        # 2. Cluster Points
        clusters = []
        current_cluster = [dynamic_points[0]]
        
        # Gap threshold in indices (approximate based on distance)
        # Using simple index gap often fails if object is far. 
        # Better: Euclidean distance between consecutive points.
        
        for k in range(1, len(dynamic_points)):
            p1 = dynamic_points[k-1]
            p2 = dynamic_points[k]
            
            dist = math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])
            
            if dist < self.config['cluster_gap_threshold']:
                current_cluster.append(p2)
            else:
                if len(current_cluster) >= self.config['min_cluster_points']:
                    clusters.append(current_cluster)
                current_cluster = [p2]
        
        if len(current_cluster) >= self.config['min_cluster_points']:
            clusters.append(current_cluster)

        # 3. Analyze Clusters
        for c in clusters:
            xs = [p['x'] for p in c]
            ys = [p['y'] for p in c]
            
            width = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
            
            # Width Check (Human Sized?)
            if self.config['human_width_min'] < width < self.config['human_width_max']:
                # Centroid
                avg_x = sum(xs) / len(xs)
                avg_y = sum(ys) / len(ys)
                
                # --- DISTANCE VALIDATION ---
                # Is this cluster close to where we expect the person?
                dist_to_expected = math.hypot(avg_x - expected_pose['x'], 
                                              avg_y - expected_pose['y'])
                
                if dist_to_expected < self.config['location_match_tolerance']:
                    self.get_logger().info(f"Human Detected at ({avg_x:.2f}, {avg_y:.2f}) (width {width:.2f}m)")
                    return True, {'x': avg_x, 'y': avg_y}
                else:
                    # Debug log: Found SOMETHING, but it's too far (maybe a box or another person)
                    self.get_logger().debug(f"Ignored object at ({avg_x:.2f}, {avg_y:.2f}) - Too far from target")

        return False, None

    # ================= Navigation =================
    def go_to_pose(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw/2)
        goal.pose.orientation.w = math.cos(yaw/2)
        
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

    # ================= Mission =================
    def mission_loop(self):
        if self.mission_active: return
        self.mission_active = True

        # 1. Setup
        init_pose = PoseStamped()
        init_pose.header.frame_id = 'map'
        init_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        init_pose.pose.position.x = self.robot_start['x']
        init_pose.pose.position.y = self.robot_start['y']
        q = self.robot_start['yaw']
        init_pose.pose.orientation.z = math.sin(q/2)
        init_pose.pose.orientation.w = math.cos(q/2)
        
        self.navigator.setInitialPose(init_pose)
        self.navigator.waitUntilNav2Active()
        
        # Wait for map
        while not self.map_received:
            self.get_logger().info("Waiting for map...")
            rclpy.spin_once(self, timeout_sec=1.0)

        # 2. Check Person 1
        # Drive to observation point
        self.get_logger().info("--- Checking Person 1 ---")
        self.go_to_pose(1.0, -3.0, 1.57) # Stand off distance
        
        # Stop and scan
        time.sleep(1.0) 
        rclpy.spin_once(self, timeout_sec=0.5) # Flush buffers
        
        found1, loc1 = self.scan_for_human(self.person1_expected)
        if found1:
            self.get_logger().info(">>> PERSON 1 FOUND at location.")
        else:
            self.get_logger().warn(">>> PERSON 1 MISSING/MOVED.")

        # 3. Check Person 2
        self.get_logger().info("--- Checking Person 2 ---")
        self.go_to_pose(-10.0, 13.0, 1.57)
        
        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.5)
        
        found2, loc2 = self.scan_for_human(self.person2_expected)
        if found2:
            self.get_logger().info(">>> PERSON 2 FOUND at location.")
        else:
            self.get_logger().warn(">>> PERSON 2 MISSING/MOVED.")

        self.get_logger().info("Mission Complete.")
        exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = WarehouseControllerStrict()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()