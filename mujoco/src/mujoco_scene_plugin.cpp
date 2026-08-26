#include <chrono>
#include <atomic>
#include <cmath>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <mujoco_ros2_control_msgs/srv/reset_world.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>

namespace mujoco_scene_plugin {

class MuJoCoScenePlugin final
    : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase {
  public:
    bool init(rclcpp::Node::SharedPtr node, const mjModel *model, mjData *data) override {
        node_ = node;
        model_ = model;

        cube_joint_id_ = mj_name2id(model, mjOBJ_JOINT, "cube_free_joint");
        if (cube_joint_id_ < 0 || model->jnt_type[cube_joint_id_] != mjJNT_FREE) {
            RCLCPP_ERROR(node_->get_logger(), "MuJoCo free joint 'cube_free_joint' was not found");
            return false;
        }
        cube_qpos_adr_ = model->jnt_qposadr[cube_joint_id_];

        const int tray_geom_id = mj_name2id(model, mjOBJ_GEOM, "left_tray");
        const int cube_geom_id = mj_name2id(model, mjOBJ_GEOM, "cube_0_geom");
        if (tray_geom_id < 0 || cube_geom_id < 0 ||
            model->geom_type[tray_geom_id] != mjGEOM_BOX ||
            model->geom_type[cube_geom_id] != mjGEOM_BOX) {
            RCLCPP_ERROR(
                node_->get_logger(),
                "MJCF must define box geoms 'left_tray' and 'cube_0_geom'");
            return false;
        }

        const int tray_pos_adr = 3 * tray_geom_id;
        const int tray_size_adr = 3 * tray_geom_id;
        const int cube_size_adr = 3 * cube_geom_id;
        tray_center_x_ = model->geom_pos[tray_pos_adr];
        tray_center_y_ = model->geom_pos[tray_pos_adr + 1];
        tray_half_x_ = model->geom_size[tray_size_adr];
        tray_half_y_ = model->geom_size[tray_size_adr + 1];
        cube_half_x_ = model->geom_size[cube_size_adr];
        cube_half_y_ = model->geom_size[cube_size_adr + 1];
        const double tray_top_z =
            model->geom_pos[tray_pos_adr + 2] + model->geom_size[tray_size_adr + 2];
        const double cube_half_z = model->geom_size[cube_size_adr + 2];
        cube_spawn_z_ = tray_top_z + cube_half_z + kDropHeight;
        RCLCPP_INFO(
            node_->get_logger(),
            "Spawn geometry: tray_top_z=%.3f cube_half_z=%.3f spawn_z=%.3f",
            tray_top_z, cube_half_z, cube_spawn_z_);

        spawn_key_id_ = mj_name2id(model, mjOBJ_KEY, "spawn");
        home_key_id_ = mj_name2id(model, mjOBJ_KEY, "home");
        if (spawn_key_id_ < 0 || home_key_id_ < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Keyframes 'home' and 'spawn' must be defined in the MJCF");
            return false;
        }

        static const char *const arm_joint_names[] = {
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
        for (const char *name : arm_joint_names) {
            const int joint_id = mj_name2id(model, mjOBJ_JOINT, name);
            if (joint_id < 0) {
                RCLCPP_ERROR(node_->get_logger(), "Home joint '%s' was not found", name);
                return false;
            }
            home_joint_names_.emplace_back(name);
            home_joint_positions_.push_back(
                model->key_qpos[home_key_id_ * model->nq + model->jnt_qposadr[joint_id]]);
        }

        latest_qpos_.resize(model->nq);
        latest_qvel_.resize(model->nv);
        latest_ctrl_.resize(model->nu);
        if (data != nullptr) {
            mju_copy(latest_qpos_.data(), data->qpos, model->nq);
            mju_copy(latest_qvel_.data(), data->qvel, model->nv);
            mju_copy(latest_ctrl_.data(), data->ctrl, model->nu);
        }

        reset_world_client_ = node_->create_client<mujoco_ros2_control_msgs::srv::ResetWorld>(
            "/mujoco_ros2_control_node/reset_world");
        home_trajectory_pub_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory_controller/joint_trajectory", 10);

        spawn_service_ = node_->create_service<std_srvs::srv::Trigger>(
            "/spawn_cube",
            [this](
                const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
                std::lock_guard<std::mutex> lock(service_mutex_);
                std::uniform_real_distribution<double> x_dist(
                    tray_center_x_ - tray_half_x_ + cube_half_x_ + kTrayMargin,
                    tray_center_x_ + tray_half_x_ - cube_half_x_ - kTrayMargin);
                std::uniform_real_distribution<double> y_dist(
                    tray_center_y_ - tray_half_y_ + cube_half_y_ + kTrayMargin,
                    tray_center_y_ + tray_half_y_ - cube_half_y_ - kTrayMargin);
                std::uniform_real_distribution<double> yaw_dist(-kPi, kPi);

                const double x = x_dist(random_);
                const double y = y_dist(random_);
                const double yaw = yaw_dist(random_);
                if (!prepare_spawn_keyframe(x, y, yaw)) {
                    response->success = false;
                    response->message = "latest MuJoCo state is unavailable";
                    return;
                }
                response->success = request_reset_world("spawn");
                response->message = response->success ?
                    "cube spawn requested (robot state preserved)" : "reset_world unavailable";
            });

        reset_service_ = node_->create_service<std_srvs::srv::Trigger>(
            "/reset",
            [this](
                const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
                std::lock_guard<std::mutex> lock(service_mutex_);
                response->success = request_reset_world("home");
                if (response->success) {
                    publish_home_trajectory();
                }
                response->message = response->success ? "reset requested" : "reset_world unavailable";
            });

        RCLCPP_INFO(node_->get_logger(), "MuJoCo scene plugin ready (reset + state-preserving cube spawn)");
        return true;
    }

    void update(const mjModel *model, mjData *data) override {
        if (data == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(state_mutex_);
        mju_copy(latest_qpos_.data(), data->qpos, model->nq);
        mju_copy(latest_qvel_.data(), data->qvel, model->nv);
        mju_copy(latest_ctrl_.data(), data->ctrl, model->nu);
    }

    void cleanup() override {
        alive_->store(false);
        home_trajectory_pub_.reset();
        spawn_service_.reset();
        reset_service_.reset();
        reset_world_client_.reset();
        node_.reset();
    }

    private:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kTrayMargin = 0.01;
    static constexpr double kDropHeight = 0.02;

    bool prepare_spawn_keyframe(double x, double y, double yaw) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (latest_qpos_.size() != static_cast<size_t>(model_->nq) ||
            latest_qvel_.size() != static_cast<size_t>(model_->nv) ||
            latest_ctrl_.size() != static_cast<size_t>(model_->nu)) {
            return false;
        }

        mjtNum *key_qpos = model_->key_qpos + spawn_key_id_ * model_->nq;
        mjtNum *key_qvel = model_->key_qvel + spawn_key_id_ * model_->nv;
        mjtNum *key_ctrl = model_->key_ctrl + spawn_key_id_ * model_->nu;
        mju_copy(key_qpos, latest_qpos_.data(), model_->nq);
        mju_copy(key_qvel, latest_qvel_.data(), model_->nv);
        mju_copy(key_ctrl, latest_ctrl_.data(), model_->nu);

        key_qpos[cube_qpos_adr_] = x;
        key_qpos[cube_qpos_adr_ + 1] = y;
        key_qpos[cube_qpos_adr_ + 2] = cube_spawn_z_;
        key_qpos[cube_qpos_adr_ + 3] = std::cos(0.5 * yaw);
        key_qpos[cube_qpos_adr_ + 4] = 0.0;
        key_qpos[cube_qpos_adr_ + 5] = 0.0;
        key_qpos[cube_qpos_adr_ + 6] = std::sin(0.5 * yaw);
        return true;
    }

    bool request_reset_world(const std::string &keyframe) {
        if (!reset_world_client_->wait_for_service(std::chrono::seconds(2))) {
            RCLCPP_ERROR(node_->get_logger(), "reset_world service not available");
            return false;
        }
        auto request = std::make_shared<mujoco_ros2_control_msgs::srv::ResetWorld::Request>();
        request->keyframe = keyframe;
        auto alive = alive_;
        std::weak_ptr<rclcpp::Node> weak_node = node_;
        reset_world_client_->async_send_request(
            request,
            [alive, weak_node](rclcpp::Client<mujoco_ros2_control_msgs::srv::ResetWorld>::SharedFuture future) {
                if (!alive->load()) {
                    return;
                }
                const auto result = future.get();
                if (!result->success) {
                    if (auto node = weak_node.lock()) {
                        RCLCPP_WARN(node->get_logger(), "reset_world failed: %s", result->message.c_str());
                    }
                }
            });
        return true;
    }

    void publish_home_trajectory() {
        if (!home_trajectory_pub_) {
            return;
        }
        trajectory_msgs::msg::JointTrajectory trajectory;
        trajectory.joint_names = home_joint_names_;
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = home_joint_positions_;
        point.time_from_start.sec = 1;
        trajectory.points.push_back(point);
        home_trajectory_pub_->publish(trajectory);
    }

    rclcpp::Node::SharedPtr node_;
    const mjModel *model_{nullptr};
    int cube_joint_id_{-1};
    int cube_qpos_adr_{-1};
    int spawn_key_id_{-1};
    int home_key_id_{-1};
    std::vector<std::string> home_joint_names_;
    std::vector<double> home_joint_positions_;
    double tray_center_x_{0.0};
    double tray_center_y_{0.0};
    double tray_half_x_{0.0};
    double tray_half_y_{0.0};
    double cube_half_x_{0.0};
    double cube_half_y_{0.0};
    double cube_spawn_z_{0.0};
    std::vector<mjtNum> latest_qpos_;
    std::vector<mjtNum> latest_qvel_;
    std::vector<mjtNum> latest_ctrl_;
    std::mutex state_mutex_;
    std::shared_ptr<std::atomic_bool> alive_{std::make_shared<std::atomic_bool>(true)};
    std::mt19937 random_{std::random_device{}()};
    std::mutex service_mutex_;
    rclcpp::Client<mujoco_ros2_control_msgs::srv::ResetWorld>::SharedPtr reset_world_client_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr home_trajectory_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr spawn_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
};

} // namespace mujoco_scene_plugin

PLUGINLIB_EXPORT_CLASS(
    mujoco_scene_plugin::MuJoCoScenePlugin,
    mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
