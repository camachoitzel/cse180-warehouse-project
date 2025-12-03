#ifndef HUMAN_DETECTOR_NODE_HPP
#define HUMAN_DETECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace human_detector
{

// Structure to represent a 2D point
struct Point2D {
    double x;
    double y;
    
    Point2D() : x(0.0), y(0.0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
    
    double distance_to(const Point2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx*dx + dy*dy);
    }
};

// Structure to represent a detected obstacle
struct Obstacle {
    Point2D position;
    double radius;
    int num_points;
    int confidence;
    
    Obstacle() : radius(0.0), num_points(0), confidence(1) {}
};

// Structure to represent a waypoint
struct Waypoint {
    double x;
    double y;
    double yaw;
    Point2D target_human;
    
    Waypoint(double x_, double y_, double yaw_) 
        : x(x_), y(y_), yaw(yaw_) {}
};

class HumanDetectorNode : public rclcpp::Node
{
public:
    HumanDetectorNode();
    ~HumanDetectorNode() = default;
    
    void run();

private:
    // Callback functions
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    
    // Phase 1: Map analysis
    std::vector<Point2D> find_humans_in_map();
    std::vector<std::vector<Point2D>> find_clusters_in_map(
        const std::vector<Point2D>& occupied_cells, double eps);
    
    // Phase 2: Navigation
    bool navigate_to_pose(double x, double y, double yaw);
    geometry_msgs::msg::PoseStamped create_pose_stamped(double x, double y, double yaw);
    double get_yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q);
    
    // Phase 3: Verification
    std::vector<Waypoint> generate_observation_waypoints(
        const std::vector<Point2D>& human_locations);
    bool is_human_still_there(const Point2D& expected_location);
    
    // Phase 4: Exploration
    std::vector<Waypoint> generate_exploration_waypoints();
    std::vector<Obstacle> detect_obstacles_in_scan();
    std::vector<Obstacle> explore_and_find_humans();
    bool is_obstacle_in_original_map(const Point2D& position);
    std::vector<Obstacle> cluster_obstacle_detections(
        const std::vector<Obstacle>& obstacles);
    
    // Utility functions
    std::vector<Point2D> scan_to_cartesian();
    Point2D transform_to_map_frame(const Point2D& point_robot_frame);
    void publish_marker(const Point2D& position, int id, 
                       float r, float g, float b, const std::string& ns);
    
    // ROS2 subscribers
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    
    // ROS2 publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    // TF2 for coordinate transformations
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Data storage
    nav_msgs::msg::OccupancyGrid::SharedPtr map_data_;
    geometry_msgs::msg::Pose current_pose_;
    sensor_msgs::msg::LaserScan::SharedPtr current_scan_;
    
    // State flags
    bool map_received_;
    bool pose_received_;
    bool scan_received_;
    
    // Parameters
    double human_min_radius_;
    double human_max_radius_;
    double observation_distance_;
    double detection_threshold_;
    double exploration_spacing_;
};

} // namespace human_detector

#endif // HUMAN_DETECTOR_NODE_HPP