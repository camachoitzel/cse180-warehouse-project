#include "human_detector_cpp/human_detector_node.hpp"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace human_detector
{

HumanDetectorNode::HumanDetectorNode()
    : Node("human_detector_cpp"),
      map_received_(false),
      pose_received_(false),
      scan_received_(false),
      human_min_radius_(0.15),
      human_max_radius_(0.6),
      observation_distance_(2.0),
      detection_threshold_(0.5),
      exploration_spacing_(3.0)
{
    // Initialize TF2
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // Create subscribers
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 10,
        std::bind(&HumanDetectorNode::map_callback, this, std::placeholders::_1));
    
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/amcl_pose", 10,
        std::bind(&HumanDetectorNode::pose_callback, this, std::placeholders::_1));
    
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        std::bind(&HumanDetectorNode::scan_callback, this, std::placeholders::_1));
    
    // Create publishers
    goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10);
    
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/detected_humans", 10);
    
    RCLCPP_INFO(this->get_logger(), "Human Detector Node Started!");
}

void HumanDetectorNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    if (!map_received_) {
        map_data_ = msg;
        map_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Map received! Size: %dx%d, Resolution: %.3fm",
                    msg->info.width, msg->info.height, msg->info.resolution);
    }
}

void HumanDetectorNode::pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    current_pose_ = msg->pose.pose;
    pose_received_ = true;
}

void HumanDetectorNode::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    current_scan_ = msg;
    scan_received_ = true;
}

// Utility: Get yaw angle from quaternion
double HumanDetectorNode::get_yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(tf_q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
}

// Utility: Create a PoseStamped message
geometry_msgs::msg::PoseStamped HumanDetectorNode::create_pose_stamped(
    double x, double y, double yaw)
{
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = this->now();
    
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = 0.0;
    
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    pose.pose.orientation = tf2::toMsg(q);
    
    return pose;
}

// PHASE 1: Find humans in map using clustering
std::vector<Point2D> HumanDetectorNode::find_humans_in_map()
{
    if (!map_data_) {
        RCLCPP_WARN(this->get_logger(), "No map data available!");
        return {};
    }
    
    int width = map_data_->info.width;
    int height = map_data_->info.height;
    double resolution = map_data_->info.resolution;
    double origin_x = map_data_->info.origin.position.x;
    double origin_y = map_data_->info.origin.position.y;
    
    // Find all occupied cells
    std::vector<Point2D> occupied_cells;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            if (map_data_->data[index] == 100) {  // Occupied
                Point2D cell;
                cell.x = origin_x + x * resolution;
                cell.y = origin_y + y * resolution;
                occupied_cells.push_back(cell);
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Found %zu occupied cells", occupied_cells.size());
    
    if (occupied_cells.empty()) {
        return {};
    }
    
    // Cluster occupied cells
    double cluster_distance = resolution * 2.0;  // Two cells apart = same cluster
    auto clusters = find_clusters_in_map(occupied_cells, cluster_distance);
    
    RCLCPP_INFO(this->get_logger(), "Found %zu clusters", clusters.size());
    
    // Find cluster centers and filter by size
    std::vector<Point2D> human_locations;
    
    int min_cluster_size = 10;
    int max_cluster_size = 200;
    
    for (const auto& cluster : clusters) {
        if (cluster.size() >= static_cast<size_t>(min_cluster_size) && 
            cluster.size() <= static_cast<size_t>(max_cluster_size)) {
            
            // Calculate centroid
            Point2D centroid;
            for (const auto& point : cluster) {
                centroid.x += point.x;
                centroid.y += point.y;
            }
            centroid.x /= cluster.size();
            centroid.y /= cluster.size();
            
            human_locations.push_back(centroid);
            
            RCLCPP_INFO(this->get_logger(), 
                       "Human found at x=%.2f, y=%.2f (cluster size: %zu cells)",
                       centroid.x, centroid.y, cluster.size());
        }
    }
    
    return human_locations;
}

// Simple DBSCAN-like clustering
std::vector<std::vector<Point2D>> HumanDetectorNode::find_clusters_in_map(
    const std::vector<Point2D>& points, double eps)
{
    std::vector<bool> visited(points.size(), false);
    std::vector<std::vector<Point2D>> clusters;
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (visited[i]) continue;
        
        // Start new cluster
        std::vector<Point2D> cluster;
        std::vector<size_t> to_check = {i};
        visited[i] = true;
        
        while (!to_check.empty()) {
            size_t current = to_check.back();
            to_check.pop_back();
            cluster.push_back(points[current]);
            
            // Find neighbors
            for (size_t j = 0; j < points.size(); ++j) {
                if (!visited[j] && points[current].distance_to(points[j]) < eps) {
                    visited[j] = true;
                    to_check.push_back(j);
                }
            }
        }
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

// PHASE 2: Navigation (simplified - you'll integrate with Nav2 action client)
bool HumanDetectorNode::navigate_to_pose(double x, double y, double yaw)
{
    RCLCPP_INFO(this->get_logger(), "Navigating to x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
    
    // Create goal pose
    auto goal_pose = create_pose_stamped(x, y, yaw);
    
    // Publish goal (in real implementation, use Nav2 action client)
    goal_pub_->publish(goal_pose);
    
    // Wait for navigation (simplified - should use action feedback)
    std::this_thread::sleep_for(5s);
    
    // Check if we're close to goal
    double dx = current_pose_.position.x - x;
    double dy = current_pose_.position.y - y;
    double distance = std::sqrt(dx*dx + dy*dy);
    
    if (distance < 0.5) {  // Within 50cm
        RCLCPP_INFO(this->get_logger(), "Navigation succeeded!");
        return true;
    } else {
        RCLCPP_WARN(this->get_logger(), "Navigation failed! Distance to goal: %.2fm", distance);
        return false;
    }
}

// PHASE 3: Generate observation waypoints
std::vector<Waypoint> HumanDetectorNode::generate_observation_waypoints(
    const std::vector<Point2D>& human_locations)
{
    std::vector<Waypoint> waypoints;
    
    for (const auto& human_pos : human_locations) {
        // Create 4 observation points around each human (N, S, E, W)
        std::vector<double> angles = {0.0, M_PI/2, M_PI, 3*M_PI/2};
        
        for (double angle : angles) {
            double wp_x = human_pos.x + observation_distance_ * std::cos(angle);
            double wp_y = human_pos.y + observation_distance_ * std::sin(angle);
            
            // Yaw should face toward human
            double yaw = std::atan2(human_pos.y - wp_y, human_pos.x - wp_x);
            
            Waypoint wp(wp_x, wp_y, yaw);
            wp.target_human = human_pos;
            waypoints.push_back(wp);
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Generated %zu observation waypoints", waypoints.size());
    return waypoints;
}

// PHASE 3: Check if human is at expected location
bool HumanDetectorNode::is_human_still_there(const Point2D& expected_location)
{
    if (!current_scan_ || !pose_received_) {
        RCLCPP_WARN(this->get_logger(), "Missing scan or pose data");
        return false;
    }
    
    // Calculate vector from robot to expected human location
    double dx = expected_location.x - current_pose_.position.x;
    double dy = expected_location.y - current_pose_.position.y;
    
    double expected_distance = std::sqrt(dx*dx + dy*dy);
    double robot_yaw = get_yaw_from_quaternion(current_pose_.orientation);
    
    // Angle to human in global frame
    double angle_global = std::atan2(dy, dx);
    
    // Convert to robot's local frame
    double angle_local = angle_global - robot_yaw;
    
    // Normalize to [-pi, pi]
    while (angle_local > M_PI) angle_local -= 2*M_PI;
    while (angle_local < -M_PI) angle_local += 2*M_PI;
    
    // Find corresponding laser beam
    int beam_index = static_cast<int>(
        (angle_local - current_scan_->angle_min) / current_scan_->angle_increment);
    
    if (beam_index < 0 || beam_index >= static_cast<int>(current_scan_->ranges.size())) {
        RCLCPP_WARN(this->get_logger(), "Human outside scan range");
        return false;
    }
    
    float measured_distance = current_scan_->ranges[beam_index];
    
    // Check if measurement is valid
    if (measured_distance < current_scan_->range_min || 
        measured_distance > current_scan_->range_max) {
        return false;
    }
    
    double distance_diff = std::abs(measured_distance - expected_distance);
    
    RCLCPP_INFO(this->get_logger(), 
               "Expected: %.2fm, Measured: %.2fm, Diff: %.2fm",
               expected_distance, measured_distance, distance_diff);
    
    if (distance_diff < detection_threshold_) {
        return true;  // Human still there
    } else if (measured_distance > expected_distance + detection_threshold_) {
        return false;  // Human moved (seeing past their location)
    }
    
    return false;
}

// PHASE 4: Generate exploration grid
std::vector<Waypoint> HumanDetectorNode::generate_exploration_waypoints()
{
    std::vector<Waypoint> waypoints;
    
    // Define exploration area (adjust for your warehouse)
    double x_min = -5.0, x_max = 15.0;
    double y_min = -25.0, y_max = -5.0;
    
    for (double x = x_min; x <= x_max; x += exploration_spacing_) {
        for (double y = y_min; y <= y_max; y += exploration_spacing_) {
            waypoints.push_back(Waypoint(x, y, 0.0));
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Created %zu exploration waypoints", waypoints.size());
    return waypoints;
}

// Convert laser scan to Cartesian coordinates (robot frame)
std::vector<Point2D> HumanDetectorNode::scan_to_cartesian()
{
    std::vector<Point2D> points;
    
    if (!current_scan_) return points;
    
    for (size_t i = 0; i < current_scan_->ranges.size(); ++i) {
        float range = current_scan_->ranges[i];
        
        // Skip invalid readings
        if (range < current_scan_->range_min || range > current_scan_->range_max) {
            continue;
        }
        
        // Calculate angle of this beam
        float angle = current_scan_->angle_min + i * current_scan_->angle_increment;
        
        // Convert polar to Cartesian
        Point2D point;
        point.x = range * std::cos(angle);
        point.y = range * std::sin(angle);
        points.push_back(point);
    }
    
    return points;
}

// Transform point from robot frame to map frame
Point2D HumanDetectorNode::transform_to_map_frame(const Point2D& point_robot_frame)
{
    if (!pose_received_) {
        return Point2D(0.0, 0.0);
    }
    
    // Get robot's current position and orientation
    double robot_x = current_pose_.position.x;
    double robot_y = current_pose_.position.y;
    double robot_yaw = get_yaw_from_quaternion(current_pose_.orientation);
    
    // Rotation matrix
    double cos_yaw = std::cos(robot_yaw);
    double sin_yaw = std::sin(robot_yaw);
    
    // Transform: P_map = P_robot + R * P_local
    Point2D point_map;
    point_map.x = robot_x + point_robot_frame.x * cos_yaw - point_robot_frame.y * sin_yaw;
    point_map.y = robot_y + point_robot_frame.x * sin_yaw + point_robot_frame.y * cos_yaw;
    
    return point_map;
}

// Detect obstacles in current laser scan
std::vector<Obstacle> HumanDetectorNode::detect_obstacles_in_scan()
{
    // Convert scan to Cartesian points
    auto points_robot = scan_to_cartesian();
    
    if (points_robot.size() < 10) {
        return {};
    }
    
    // Cluster points using DBSCAN-like algorithm
    double cluster_eps = 0.3;  // 30cm - points within this distance are in same cluster
    auto clusters = find_clusters_in_map(points_robot, cluster_eps);
    
    std::vector<Obstacle> obstacles;
    
    // Analyze each cluster
    for (const auto& cluster : clusters) {
        // Need at least 5 points to be a valid obstacle
        if (cluster.size() < 5) continue;
        
        // Calculate cluster center (in robot frame)
        Point2D center_robot;
        for (const auto& pt : cluster) {
            center_robot.x += pt.x;
            center_robot.y += pt.y;
        }
        center_robot.x /= cluster.size();
        center_robot.y /= cluster.size();
        
        // Calculate cluster radius (max distance from center to any point)
        double max_dist = 0.0;
        for (const auto& pt : cluster) {
            double dist = center_robot.distance_to(pt);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        
        // Filter by size - only keep human-sized objects
        // Humans are typically 0.15m to 0.6m radius
        if (max_dist >= human_min_radius_ && max_dist <= human_max_radius_) {
            Obstacle obs;
            // Transform center from robot frame to map frame
            obs.position = transform_to_map_frame(center_robot);
            obs.radius = max_dist;
            obs.num_points = cluster.size();
            obs.confidence = 1;
            obstacles.push_back(obs);
        }
    }
    
    return obstacles;
}

// Check if obstacle position matches something in the original map
bool HumanDetectorNode::is_obstacle_in_original_map(const Point2D& position)
{
    if (!map_data_) {
        return false;
    }
    
    // Get map parameters
    double resolution = map_data_->info.resolution;
    double origin_x = map_data_->info.origin.position.x;
    double origin_y = map_data_->info.origin.position.y;
    int width = map_data_->info.width;
    int height = map_data_->info.height;
    
    // Convert world coordinates to grid coordinates
    int grid_x = static_cast<int>((position.x - origin_x) / resolution);
    int grid_y = static_cast<int>((position.y - origin_y) / resolution);
    
    // Check bounds
    if (grid_x < 0 || grid_x >= width || grid_y < 0 || grid_y >= height) {
        return false;
    }
    
    // Check a small region around this point (not just single cell)
    // This accounts for slight position uncertainties
    int search_radius = 2;  // Check +/- 2 cells
    
    for (int dy = -search_radius; dy <= search_radius; ++dy) {
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            int check_x = grid_x + dx;
            int check_y = grid_y + dy;
            
            // Check bounds
            if (check_x >= 0 && check_x < width && check_y >= 0 && check_y < height) {
                int index = check_y * width + check_x;
                
                // If any cell in this region is occupied in original map
                if (map_data_->data[index] == 100) {
                    return true;  // This obstacle was in the original map
                }
            }
        }
    }
    
    return false;  // New obstacle, not in original map
}

// Cluster nearby obstacle detections into final positions
std::vector<Obstacle> HumanDetectorNode::cluster_obstacle_detections(
    const std::vector<Obstacle>& obstacles)
{
    if (obstacles.empty()) {
        return {};
    }
    
    // Extract positions from obstacles
    std::vector<Point2D> positions;
    for (const auto& obs : obstacles) {
        positions.push_back(obs.position);
    }
    
    // Cluster detections - same human seen from multiple waypoints
    // should be within 1 meter of each other
    double cluster_distance = 1.0;
    auto clusters = find_clusters_in_map(positions, cluster_distance);
    
    std::vector<Obstacle> final_obstacles;
    
    for (const auto& cluster : clusters) {
        if (cluster.empty()) continue;
        
        Obstacle final_obs;
        
        // Calculate average position of all detections in this cluster
        for (const auto& pt : cluster) {
            final_obs.position.x += pt.x;
            final_obs.position.y += pt.y;
        }
        final_obs.position.x /= cluster.size();
        final_obs.position.y /= cluster.size();
        
        // Confidence = number of times we detected this human
        final_obs.confidence = cluster.size();
        final_obs.radius = 0.3;  // Typical human radius
        final_obs.num_points = 0;
        
        final_obstacles.push_back(final_obs);
        
        RCLCPP_INFO(this->get_logger(),
                   "Final human position: x=%.2f, y=%.2f (confidence: %d detections)",
                   final_obs.position.x, final_obs.position.y, final_obs.confidence);
    }
    
    return final_obstacles;
}

// PHASE 4: Systematic exploration to find moved humans
std::vector<Obstacle> HumanDetectorNode::explore_and_find_humans()
{
    RCLCPP_INFO(this->get_logger(), "Starting systematic exploration...");
    
    // Generate grid of waypoints to explore
    auto waypoints = generate_exploration_waypoints();
    
    // Storage for all detected obstacles
    std::vector<Obstacle> all_obstacles;
    
    // Visit each waypoint
    for (size_t i = 0; i < waypoints.size(); ++i) {
        const auto& wp = waypoints[i];
        
        RCLCPP_INFO(this->get_logger(), 
                   "Exploring waypoint %zu/%zu: x=%.2f, y=%.2f",
                   i+1, waypoints.size(), wp.x, wp.y);
        
        // Navigate to this waypoint
        bool success = navigate_to_pose(wp.x, wp.y, wp.yaw);
        
        if (!success) {
            RCLCPP_WARN(this->get_logger(), "Failed to reach waypoint %zu, skipping", i+1);
            continue;
        }
        
        // Wait for robot to stabilize and get fresh scan
        std::this_thread::sleep_for(500ms);
        
        // Detect obstacles at this location
        auto obstacles = detect_obstacles_in_scan();
        
        RCLCPP_INFO(this->get_logger(), "Detected %zu potential obstacles", obstacles.size());
        
        // Filter: keep only obstacles NOT in original map
        for (const auto& obs : obstacles) {
            if (!is_obstacle_in_original_map(obs.position)) {
                all_obstacles.push_back(obs);
                RCLCPP_INFO(this->get_logger(),
                           "New obstacle found at x=%.2f, y=%.2f, radius=%.2fm",
                           obs.position.x, obs.position.y, obs.radius);
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), 
               "Exploration complete. Total new obstacles detected: %zu",
               all_obstacles.size());
    
    // Cluster all detections to get final human positions
    // (Same human might be detected from multiple waypoints)
    auto final_positions = cluster_obstacle_detections(all_obstacles);
    
    return final_positions;
}

// Optional: Publish visualization markers (useful for RViz)
void HumanDetectorNode::publish_marker(const Point2D& position, int id, 
                                       float r, float g, float b, 
                                       const std::string& ns)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    marker.pose.position.x = position.x;
    marker.pose.position.y = position.y;
    marker.pose.position.z = 0.5;  // Half meter above ground
    
    marker.pose.orientation.w = 1.0;
    
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 1.0;
    
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;
    
    marker.lifetime = rclcpp::Duration::from_seconds(0);  // Forever
    
    // You'd need to create a MarkerArray and publish
    // This is just the helper function
}

// MAIN RUN FUNCTION - Orchestrates the entire mission
void HumanDetectorNode::run()
{
    RCLCPP_INFO(this->get_logger(), "Waiting for all data...");
    
    // Wait for all necessary data to arrive
    rclcpp::Rate rate(2);  // Check at 2 Hz
    while (rclcpp::ok() && (!map_received_ || !pose_received_ || !scan_received_)) {
        rclcpp::spin_some(this->get_node_base_interface());
        rate.sleep();
        
        // Provide status updates
        if (!map_received_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                "Waiting for map...");
        }
        if (!pose_received_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Waiting for pose...");
        }
        if (!scan_received_) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Waiting for scan...");
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "All data received! Starting mission...");
    
    // ===================================================================
    // PHASE 1: Find expected human locations from the original map
    // ===================================================================
    auto expected_humans = find_humans_in_map();
    
    if (expected_humans.empty()) {
        RCLCPP_ERROR(this->get_logger(), 
                    "No humans found in map! Check your map file or clustering parameters.");
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), 
               "Found %zu expected human locations in map", 
               expected_humans.size());
    
    // ===================================================================
    // PHASE 2 & 3: Navigate to observation points and verify each human
    // ===================================================================
    auto observation_waypoints = generate_observation_waypoints(expected_humans);
    std::vector<Point2D> humans_moved;
    
    RCLCPP_INFO(this->get_logger(), 
               "Checking %zu observation points...", 
               observation_waypoints.size());
    
    for (size_t i = 0; i < observation_waypoints.size(); ++i) {
        const auto& wp = observation_waypoints[i];
        
        RCLCPP_INFO(this->get_logger(),
                   "\nChecking observation point %zu/%zu",
                   i+1, observation_waypoints.size());
        
        RCLCPP_INFO(this->get_logger(),
                   "Target human at: x=%.2f, y=%.2f",
                   wp.target_human.x, wp.target_human.y);
        
        // Navigate to observation point
        bool success = navigate_to_pose(wp.x, wp.y, wp.yaw);
        
        if (!success) {
            RCLCPP_WARN(this->get_logger(), 
                       "Could not reach observation point, skipping");
            continue;
        }
        
        // Wait for stable sensor data
        std::this_thread::sleep_for(1s);
        
        // Check if human is at expected location
        bool is_present = is_human_still_there(wp.target_human);
        
        if (!is_present) {
            RCLCPP_WARN(this->get_logger(), 
                       "Human at (%.2f, %.2f) has MOVED!",
                       wp.target_human.x, wp.target_human.y);
            
            // Add to list of moved humans (avoid duplicates)
            bool already_added = false;
            for (const auto& moved : humans_moved) {
                if (moved.distance_to(wp.target_human) < 0.1) {
                    already_added = true;
                    break;
                }
            }
            if (!already_added) {
                humans_moved.push_back(wp.target_human);
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "Human still in original position");
        }
    }
    
    // ===================================================================
    // PHASE 4: If any humans moved, explore to find their new positions
    // ===================================================================
    if (!humans_moved.empty()) {
        RCLCPP_INFO(this->get_logger(),
                   "\n========================================");
        RCLCPP_INFO(this->get_logger(),
                   "%zu human(s) have moved! Starting search...",
                   humans_moved.size());
        RCLCPP_INFO(this->get_logger(),
                   "========================================\n");
        
        // Perform systematic exploration
        auto new_positions = explore_and_find_humans();
        
        // ===================================================================
        // FINAL REPORT
        // ===================================================================
        RCLCPP_INFO(this->get_logger(), "\n");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "           FINAL RESULTS");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Humans that moved: %zu", humans_moved.size());
        RCLCPP_INFO(this->get_logger(), "New positions found: %zu", new_positions.size());
        RCLCPP_INFO(this->get_logger(), "----------------------------------------");
        
        if (new_positions.empty()) {
            RCLCPP_WARN(this->get_logger(), 
                       "No new positions found! Humans may be in unexplored areas.");
        } else {
            for (size_t i = 0; i < new_positions.size(); ++i) {
                const auto& pos = new_positions[i];
                RCLCPP_INFO(this->get_logger(),
                           "Human %zu:", i+1);
                RCLCPP_INFO(this->get_logger(),
                           "  Position: x=%.2f, y=%.2f",
                           pos.position.x, pos.position.y);
                RCLCPP_INFO(this->get_logger(),
                           "  Confidence: %d detections",
                           pos.confidence);
                RCLCPP_INFO(this->get_logger(),
                           "  Radius: %.2fm", pos.radius);
            }
        }
        RCLCPP_INFO(this->get_logger(), "========================================\n");
        
    } else {
        RCLCPP_INFO(this->get_logger(), "\n");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "All humans are in their original positions!");
        RCLCPP_INFO(this->get_logger(), "No exploration needed.");
        RCLCPP_INFO(this->get_logger(), "========================================\n");
    }
    
    RCLCPP_INFO(this->get_logger(), "Mission complete!");
}

} // namespace human_detector