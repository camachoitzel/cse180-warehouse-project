#!/usr/bin/env python3
"""
Map-Aware Warehouse Human Detection (Vantage Point Fix)
=======================================================
FIX: 
1. Separates 'Target Location' from 'Robot Parking Spot'.
2. Robot parks 1.5m away to ensure target is within Lidar range.
3. Strict static filtering to ignore shelves.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from nav2_simple_commander.robot_navigator import BasicNavigator
import math
import numpy as np
import time

class WarehouseInspector(Node):
    def __init__(self):
        super().__init__('warehouse_inspector')
        
        # --- CONFIGURATION ---
        self.config = {
            # Person 1 (Standing)
            'p1_loc': {'x': 1.0, 'y': -1.0},
            'p1_view': {'x': 1.0, 'y': -0.85, 'yaw': 1.57}, # Look North at them
            
            # Person 2 (Walking)
            'p2_loc': {'x': -12.0, 'y': 15.0},
            'p2_view': {'x': -12.0, 'y': 13.0, 'yaw': 1.57}, # Look towards them
            
            # Filters
            'static_buffer': 0.20,       # Ignore points within 20cm of map obstacles
            'lidar_min_range': 0.20,     # Ignore noise/self-collisions close to robot
            'human_width_min': 0.10,
            'human_width_max': 0.80,     # Humans aren't wider than 80cm usually
            'match_tolerance': 1.5,      # Allow detection within 1.5m of expected spot
        }

        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.map_received = False

        # Navigation & TF
        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, 
                         durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.get_logger().info("Inspector Ready. Waiting for Map...")

    def map_cb(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        self.map_received = True

    def scan_cb(self, msg):
        self.latest_scan = msg

    # --- GEOMETRY HELPERS ---
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw
        except:
            return None

    def is_static(self, wx, wy):
        """Checks if world point (wx, wy) is inside/near a map obstacle."""
        if not self.map_received: return False
        
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Buffer check (dense window)
        buf = int(self.config['static_buffer'] / self.map_info.resolution)
        for r in range(my - buf, my + buf + 1):
            for c in range(mx - buf, mx + buf + 1):
                if 0 <= r < self.map_info.height and 0 <= c < self.map_info.width:
                    if self.map_data[r, c] > 50: # Occupied
                        return True
        return False

    def check_location(self, target_name, expected_loc):
        """Scans for a human cluster near expected_loc."""
        if not self.latest_scan: return False

        rx, ry, ryaw = self.get_robot_pose()
        scan = self.latest_scan
        points = []

        # 1. Filter Scan Data
        for i, r in enumerate(scan.ranges):
            if r < self.config['lidar_min_range'] or r > 5.0 or not math.isfinite(r):
                continue

            # Project to World
            angle = ryaw + scan.angle_min + (i * scan.angle_increment)
            wx = rx + r * math.cos(angle)
            wy = ry + r * math.sin(angle)

            # Filter Static Map Obstacles
            if not self.is_static(wx, wy):
                points.append({'x': wx, 'y': wy})

        if len(points) < 5:
            self.get_logger().warn(f"{target_name}: No dynamic points found (Clean floor).")
            return False

        # 2. Cluster Points (Euclidean Distance Clustering)
        clusters = []
        current = [points[0]]
        for i in range(1, len(points)):
            dist = math.hypot(points[i]['x'] - points[i-1]['x'], 
                              points[i]['y'] - points[i-1]['y'])
            if dist < 0.20: # 20cm gap max
                current.append(points[i])
            else:
                if len(current) > 3: clusters.append(current)
                current = [points[i]]
        if len(current) > 3: clusters.append(current)

        # 3. Analyze Clusters
        for c in clusters:
            xs = [p['x'] for p in c]
            ys = [p['y'] for p in c]
            width = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
            
            # Check Width
            if self.config['human_width_min'] < width < self.config['human_width_max']:
                cx = sum(xs)/len(xs)
                cy = sum(ys)/len(ys)
                
                # Check Distance to Expected Location
                err = math.hypot(cx - expected_loc['x'], cy - expected_loc['y'])
                if err < self.config['match_tolerance']:
                    self.get_logger().info(f"✓ FOUND {target_name} at ({cx:.2f}, {cy:.2f}) [Width: {width:.2f}m]")
                    return True
        
        self.get_logger().warn(f"✗ {target_name} MOVED (No matching object found).")
        return False

    def run(self):
        # Initial Pose
        init = PoseStamped()
        init.header.frame_id = 'map'
        init.header.stamp = self.navigator.get_clock().now().to_msg()
        init.pose.position.x = 2.12
        init.pose.position.y = -21.3
        init.pose.orientation.z = 0.707
        init.pose.orientation.w = 0.707
        self.navigator.setInitialPose(init)
        self.navigator.waitUntilNav2Active()

        # --- MISSION START ---
        
        # 1. PERSON 1
        self.get_logger().info(f"Navigating to P1 Viewpoint: {self.config['p1_view']}")
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = self.config['p1_view']['x']
        goal.pose.position.y = self.config['p1_view']['y']
        goal.pose.orientation.z = math.sin(self.config['p1_view']['yaw']/2)
        goal.pose.orientation.w = math.cos(self.config['p1_view']['yaw']/2)
        
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete(): 
            rclpy.spin_once(self, timeout_sec=0.1)
        
        time.sleep(1.0) # Stabilize
        rclpy.spin_once(self, timeout_sec=0.5) # Flush scan buffer
        self.check_location("Person 1", self.config['p1_loc'])

        # 2. PERSON 2
        self.get_logger().info(f"Navigating to P2 Viewpoint: {self.config['p2_view']}")
        goal.pose.position.x = self.config['p2_view']['x']
        goal.pose.position.y = self.config['p2_view']['y']
        goal.pose.orientation.z = math.sin(self.config['p2_view']['yaw']/2)
        goal.pose.orientation.w = math.cos(self.config['p2_view']['yaw']/2)
        
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.5)
        self.check_location("Person 2", self.config['p2_loc'])

def main():
    rclpy.init()
    node = WarehouseInspector()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()