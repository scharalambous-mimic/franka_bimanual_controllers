// Copyright (c) 2019 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_bimanual_controllers/bimanual_cartesian_impedance_controller.h>

#include <cmath>
#include <functional>
#include <memory>

#include <controller_interface/controller_base.h>
#include <eigen_conversions/eigen_msg.h>
#include <franka/robot_state.h>
#include <franka_bimanual_controllers/pseudo_inversion.h>
#include <franka_bimanual_controllers/franka_model.h>
#include <franka_hw/trigger_rate.h>
#include <franka_msgs/ErrorRecoveryActionGoal.h>
#include <geometry_msgs/PoseStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include "sensor_msgs/JointState.h"
#include <std_srvs/SetBool.h>
namespace franka_bimanual_controllers {

bool BiManualCartesianImpedanceControl::initArm(
    hardware_interface::RobotHW* robot_hw,
    const std::string& arm_id,
    const std::vector<std::string>& joint_names) {
  FrankaDataContainer arm_data;
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Error getting model interface from hardware");
    return false;
  }
  try {
    arm_data.model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Exception getting model handle from "
        "interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Error getting state interface from hardware");
    return false;
  }
  try {
    arm_data.state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Exception getting state handle from "
        "interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Error getting effort joint interface from "
        "hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      arm_data.joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "BiManualCartesianImpedanceControl: Exception getting joint handles: "
          << ex.what());
      return false;
    }
  }

  arm_data.position_d_.setZero();
  arm_data.orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  arm_data.cartesian_stiffness_.setZero();
  arm_data.cartesian_damping_.setZero();
  arm_data.force_torque.setZero();

  arms_data_.emplace(std::make_pair(arm_id, std::move(arm_data)));

  return true;
}

bool BiManualCartesianImpedanceControl::init(hardware_interface::RobotHW* robot_hw,
                                                      ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;


  if (!node_handle.getParam("left/arm_id", left_arm_id_)) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Could not read parameter left_arm_id_");
    return false;
  }
  std::vector<std::string> left_joint_names;
  if (!node_handle.getParam("left/joint_names", left_joint_names) || left_joint_names.size() != 7) {
    ROS_ERROR(
        "BiManualCartesianImpedanceControl: Invalid or no left_joint_names parameters "
        "provided, "
        "aborting controller init!");
    return false;
  }

  if (!node_handle.getParam("right/arm_id", right_arm_id_)) {
    ROS_ERROR_STREAM(
        "BiManualCartesianImpedanceControl: Could not read parameter right_arm_id_");
    return false;
  }

  std::vector<std::string> right_joint_names;
  if (!node_handle.getParam("right/joint_names", right_joint_names) ||
      right_joint_names.size() != 7) {
    ROS_ERROR(
        "BiManualCartesianImpedanceControl: Invalid or no right_joint_names parameters "
        "provided, "
        "aborting controller init!");
    return false;
  }

  bool left_success = initArm(robot_hw, left_arm_id_, left_joint_names);
  bool right_success = initArm(robot_hw, right_arm_id_, right_joint_names);

  sub_equilibrium_pose_right_ = node_handle.subscribe(
      "panda_right_equilibrium_pose", 20, &BiManualCartesianImpedanceControl::equilibriumPoseCallback_right, this,
      ros::TransportHints().reliable().tcpNoDelay());
  sub_equilibrium_pose_left_ = node_handle.subscribe(
      "panda_left_equilibrium_pose", 20, &BiManualCartesianImpedanceControl::equilibriumPoseCallback_left, this,
      ros::TransportHints().reliable().tcpNoDelay());

  sub_nullspace_right_ = node_handle.subscribe(
    "panda_right_nullspace", 20, &BiManualCartesianImpedanceControl::equilibriumConfigurationCallback_right, this,
    ros::TransportHints().reliable().tcpNoDelay());

  sub_nullspace_left_ = node_handle.subscribe(
    "panda_left_nullspace", 20, &BiManualCartesianImpedanceControl::equilibriumConfigurationCallback_left, this,
    ros::TransportHints().reliable().tcpNoDelay());

  sub_equilibrium_distance_ = node_handle.subscribe(
        "equilibrium_distance", 20, &BiManualCartesianImpedanceControl::equilibriumPoseCallback_relative, this,
        ros::TransportHints().reliable().tcpNoDelay());


  pub_right = node_handle.advertise<geometry_msgs::PoseStamped>("panda_right_cartesian_pose", 1);

  pub_left = node_handle.advertise<geometry_msgs::PoseStamped>("panda_left_cartesian_pose", 1);

  pub_force_torque_right= node_handle.advertise<geometry_msgs::WrenchStamped>("/force_torque_right_ext",1);
  pub_force_torque_left= node_handle.advertise<geometry_msgs::WrenchStamped>("/force_torque_left_ext",1);

  pub_error_recovery_ = node_handle.advertise<franka_msgs::ErrorRecoveryActionGoal>("/panda_dual/error_recovery/goal", 1, true);

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle("dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<dynamic_reconfigure::Server<
      franka_combined_bimanual_controllers::dual_arm_compliance_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);

  dynamic_server_compliance_param_->setCallback(boost::bind(
      &BiManualCartesianImpedanceControl::complianceParamCallback, this, _1, _2));

        // Define variables to store parameter values
  const std::string limit_types[2] = {"lower", "upper"};

  // Read parameters from the parameter server
  for (int i = 0; i < 7; ++i) {
      for (const std::string& limit_type : limit_types) {
          std::string param_name = "/joint" + std::to_string(i + 1) + "/limit/" + limit_type;
          if (!node_handle.getParam(param_name, joint_limits[i][limit_type == "lower" ? 0 : 1])) {
              ROS_ERROR("Failed to retrieve parameter: %s", param_name.c_str());
              return 1;
          }
      }
  }

    // Stream the parameter values
    ROS_INFO("Joint limits:");
    for (int i = 0; i < 7; ++i) {
        ROS_INFO("Joint %d: lower=%.4f, upper=%.4f", i + 1, joint_limits[i][0], joint_limits[i][1]);
    }

   // Advertise safety services
   safety_service_server_ = node_handle.advertiseService(
       "set_safety_state", &BiManualCartesianImpedanceControl::setSafetyCallback, this);

   // Subscribe to heartbeat topic
   heartbeat_sub_ = node_handle.subscribe(
       "collision_detection_heartbeat", 1, &BiManualCartesianImpedanceControl::heartbeatCallback, this);

   return left_success && right_success;
}

void BiManualCartesianImpedanceControl::starting(const ros::Time& time) {
startingArmLeft();
startingArmRight();

franka::RobotState initial_state_left = arms_data_.at(left_arm_id_).state_handle_->getRobotState();
prev_robot_mode_left_ = initial_state_left.robot_mode;

franka::RobotState initial_state_right = arms_data_.at(right_arm_id_).state_handle_->getRobotState();
prev_robot_mode_right_ = initial_state_right.robot_mode;

controller_state_ = NORMAL_OPERATION;

{
 std::lock_guard<std::mutex> lock(heartbeat_mutex_);
 last_heartbeat_time_ = time;
}
}

void BiManualCartesianImpedanceControl::update(const ros::Time& time,
                                                        const ros::Duration& /*period*/) {
  ros::Time current_last_heartbeat_time;
  bool initial_heartbeat_was_received = false;

  {
    // Read the flag and the time under the same lock to avoid race condition
    std::lock_guard<std::mutex> lock(heartbeat_mutex_);
    current_last_heartbeat_time = last_heartbeat_time_;
    initial_heartbeat_was_received = initial_heartbeat_received_.load(); // atomic read
  }

  // Check heartbeat only if the initial one has been received
  if (initial_heartbeat_was_received) { // Use the value read under the lock
     double time_diff = (time - current_last_heartbeat_time).toSec();
      if (time_diff > 0.5) { 
        if (is_safe_.load()) { // atomic read
           ROS_INFO("Heartbeat timed out. Setting controller to UNSAFE. Current time: %f, Last heartbeat: %f", 
                    time.toSec(), current_last_heartbeat_time.toSec());
            is_safe_.store(false); // atomic write
        }
      }
    }
 
  // if (!is_safe_.load()) { // atomic read
  //    // Throw an error
  //    throw std::runtime_error("Controller is not safe. Exiting update loop.");
  //  }

  if (!is_safe_.load() && controller_state_ == NORMAL_OPERATION) {
        controller_state_ = COLLISION_DETECTED;
        freezeDesiredPoses();
        ROS_INFO("Collision detected! Freezing current pose. Recycle e-stops and move arms into a non collision config to continue operation.");
  }

  // Get current robot states
  franka::RobotState robot_state_left = arms_data_.at(left_arm_id_).state_handle_->getRobotState();
  franka::RobotState robot_state_right = arms_data_.at(right_arm_id_).state_handle_->getRobotState();

  // e-stop recovery check
  bool left_needs_recovery = (prev_robot_mode_left_ == franka::RobotMode::kUserStopped && robot_state_left.robot_mode == franka::RobotMode::kIdle);
  bool right_needs_recovery = (prev_robot_mode_right_ == franka::RobotMode::kUserStopped && robot_state_right.robot_mode == franka::RobotMode::kIdle);

  if (left_needs_recovery || right_needs_recovery) {
    ROS_INFO("E-Stop cycle detected. Triggering automatic error recovery.");
    franka_msgs::ErrorRecoveryActionGoal goal_msg;
    pub_error_recovery_.publish(goal_msg);

    // if  collision had occurred, we now enter a pending state to wait for recovery to finish.
    if (controller_state_ == COLLISION_DETECTED) {
      controller_state_ = RECOVERY_PENDING;
    }
  }

  switch (controller_state_) {
    case RECOVERY_PENDING:
      // Check if both arms are active again after the recovery action was sent
      if (robot_state_left.robot_mode != franka::RobotMode::kIdle && robot_state_left.robot_mode != franka::RobotMode::kUserStopped &&
          robot_state_right.robot_mode != franka::RobotMode::kIdle && robot_state_right.robot_mode != franka::RobotMode::kUserStopped) {
            
            ROS_INFO("Both arms recovered after collision event. Resuming NORMAL_OPERATION.");
            controller_state_ = NORMAL_OPERATION;
            is_safe_.store(true); // Safety is restored

            // set desired pose to current actual pose to prevent any initial movement.
            auto& left_arm_data = arms_data_.at(left_arm_id_);
            Eigen::Affine3d tf_left(Eigen::Matrix4d::Map(robot_state_left.O_T_EE.data()));
            left_arm_data.position_d_ = tf_left.translation();
            left_arm_data.orientation_d_ = Eigen::Quaterniond(tf_left.linear());
    
            auto& right_arm_data = arms_data_.at(right_arm_id_);
            Eigen::Affine3d tf_right(Eigen::Matrix4d::Map(robot_state_right.O_T_EE.data()));
            right_arm_data.position_d_ = tf_right.translation();
            right_arm_data.orientation_d_ = Eigen::Quaterniond(tf_right.linear());
      }
      break;
    case NORMAL_OPERATION:
    case COLLISION_DETECTED:
      break;
  }


  updateArmLeft();
  updateArmRight();
  prev_robot_mode_left_ = robot_state_left.robot_mode;
  prev_robot_mode_right_ = robot_state_right.robot_mode;
}

void BiManualCartesianImpedanceControl::startingArmLeft() {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  auto& left_arm_data = arms_data_.at(left_arm_id_);

  franka::RobotState initial_state = left_arm_data.state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      left_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set target point to current state
  left_arm_data.position_d_ = initial_transform.translation();
  left_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  left_arm_data.position_d_ = initial_transform.translation();
  left_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());

  // set nullspace target configuration to initial q
  left_arm_data.q_d_nullspace_ = q_initial;
  
}

void BiManualCartesianImpedanceControl::startingArmRight() {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  franka::RobotState initial_state = right_arm_data.state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      right_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set target point to current state
  right_arm_data.position_d_ = initial_transform.translation();
  right_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  right_arm_data.position_d_ = initial_transform.translation();
  right_arm_data.orientation_d_ = Eigen::Quaterniond(initial_transform.linear());

  // set nullspace target configuration to initial q
  right_arm_data.q_d_nullspace_ = q_initial;
}

void BiManualCartesianImpedanceControl::updateArmLeft() {
  // get state variables
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  franka::RobotState robot_state_left = left_arm_data.state_handle_->getRobotState();

  //JUST FOR DEBUGGING
  ROS_INFO_THROTTLE(1.0, "Current left arm mode: %d", static_cast<int>(robot_state_left.robot_mode));


  Eigen::Vector3d position_d_target;
  Eigen::Quaterniond orientation_d_target;
  Eigen::Vector3d position_d_relative_target;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_target;

  if (controller_state_ == NORMAL_OPERATION) {
      position_d_target = left_arm_data.position_d_;
      orientation_d_target = left_arm_data.orientation_d_;
      position_d_relative_target = left_arm_data.position_d_relative_;
      q_d_nullspace_target = left_arm_data.q_d_nullspace_;
  } else { // COLLISION_DETECTED or RECOVERY_PENDING, use frozen poses
      position_d_target = frozen_pose_left_.position_d;
      orientation_d_target = frozen_pose_left_.orientation_d;
      position_d_relative_target = frozen_pose_left_.position_d_relative;
      q_d_nullspace_target = frozen_pose_left_.q_d_nullspace;
  }

  std::array<double, 49> inertia_array = left_arm_data.model_handle_->getMass();
  std::array<double, 7> coriolis_array = left_arm_data.model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      left_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  
  std::array<double, 42> jacobian_array_right =
      right_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_right(jacobian_array_right.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_left.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_left.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d( robot_state_left.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state_left.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::MatrixXd jacobian_transpose_pinv;
  franka_bimanual_controllers::pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  franka::RobotState robot_state_right = right_arm_data.state_handle_->getRobotState();
  Eigen::Affine3d transform_right(Eigen::Matrix4d::Map(robot_state_right.O_T_EE.data()));
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_right(robot_state_right.dq.data());
  Eigen::Vector3d position_right(transform_right.translation());
  // left_arm_data.position_other_arm_=position_right;
  // compute error to desired pose
  // position error
  geometry_msgs::PoseStamped msg_left;
  msg_left.pose.position.x=position[0];
  msg_left.pose.position.y=position[1];
  msg_left.pose.position.z=position[2];

  msg_left.pose.orientation.x=orientation.x();
  msg_left.pose.orientation.y=orientation.y();
  msg_left.pose.orientation.z=orientation.z();
  msg_left.pose.orientation.w=orientation.w();
  pub_left.publish(msg_left);


  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_ext(robot_state_left.tau_ext_hat_filtered.data());
  Eigen::Matrix<double, 7, 1>  tau_f;

  // Compute the value of the friction
  tau_f(0) =  FI_11/(1+exp(-FI_21*(dq(0)+FI_31))) - TAU_F_CONST_1;
  tau_f(1) =  FI_12/(1+exp(-FI_22*(dq(1)+FI_32))) - TAU_F_CONST_2;
  tau_f(2) =  FI_13/(1+exp(-FI_23*(dq(2)+FI_33))) - TAU_F_CONST_3;
  tau_f(3) =  FI_14/(1+exp(-FI_24*(dq(3)+FI_34))) - TAU_F_CONST_4;
  tau_f(4) =  FI_15/(1+exp(-FI_25*(dq(4)+FI_35))) - TAU_F_CONST_5;
  tau_f(5) =  FI_16/(1+exp(-FI_26*(dq(5)+FI_36))) - TAU_F_CONST_6;
  tau_f(6) =  FI_17/(1+exp(-FI_27*(dq(6)+FI_37))) - TAU_F_CONST_7;

  float iCutOffFrequency=10.0;
  left_arm_data.force_torque+=(-jacobian_transpose_pinv*(tau_ext-tau_f)-left_arm_data.force_torque)*(1-exp(-0.001 * 2.0 * M_PI * iCutOffFrequency));
  geometry_msgs::WrenchStamped force_torque_msg;
  force_torque_msg.wrench.force.x=left_arm_data.force_torque[0];
  force_torque_msg.wrench.force.y=left_arm_data.force_torque[1];
  force_torque_msg.wrench.force.z=left_arm_data.force_torque[2];
  force_torque_msg.wrench.torque.x=left_arm_data.force_torque[3];
  force_torque_msg.wrench.torque.y=left_arm_data.force_torque[4];
  force_torque_msg.wrench.torque.z=left_arm_data.force_torque[5];
  pub_force_torque_left.publish(force_torque_msg);

  Eigen::Matrix<double, 6, 1> error_left;
  error_left.head(3) << position - position_d_target;

  // calculate the magnitude of the position error
  double position_error_magnitude = error_left.head(3).norm();
  if (position_error_magnitude > delta_lim){
    // scale the position error to the delta_lim
    error_left.head(3) *= (delta_lim / position_error_magnitude);
  }

  Eigen::Matrix<double, 6, 1> error_relative;
  error_relative.head(3) << position - position_right;
  error_relative.tail(3).setZero();
  error_relative.head(3)<< error_relative.head(3) -position_d_relative_target;

  // calculate the magnitude of the relative position error
  double relative_error_magnitude = error_relative.head(3).norm();
  if (relative_error_magnitude > delta_lim){
    // scale the relative position error to the delta_lim
    error_relative.head(3) *= (delta_lim / relative_error_magnitude);
  }

  // orientation error
  if (orientation_d_target.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation * orientation_d_target.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error_left.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();


  // define orientation error clipping limit
  const double orientation_delta_lim = delta_lim * 3.0;
  // calculate the magnitude of the orientation error
  double orientation_error_magnitude = error_left.tail(3).norm();
  if (orientation_error_magnitude > orientation_delta_lim) {
    // scale the orientation error to the orientation_delta_lim
    error_left.tail(3) *= (orientation_delta_lim / orientation_error_magnitude);
  }

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace_left(7), tau_d_left(7), tau_joint_limit(7), null_space_error(7), tau_relative(7);

  tau_task.setZero();
  tau_d_left.setZero();
  tau_nullspace_left.setZero();
  tau_joint_limit.setZero();
  tau_relative.setZero();
  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  null_space_error.setZero();
  null_space_error(0)=(q_d_nullspace_target(0) - q(0));
  null_space_error(1)=(q_d_nullspace_target(1) - q(1));
  null_space_error(2)=(q_d_nullspace_target(2) - q(2));
  null_space_error(3)=(q_d_nullspace_target(3) - q(3));
  null_space_error(4)=(q_d_nullspace_target(4) - q(4));
  null_space_error(5)=(q_d_nullspace_target(5) - q(5));
  null_space_error(6)=(q_d_nullspace_target(6) - q(6));
  // Cartesian PD control with damping ratio = 1
  tau_task << jacobian.transpose() * (-left_arm_data.cartesian_stiffness_ * error_left -
                                      left_arm_data.cartesian_damping_ * (jacobian * dq)); 
  // nullspace PD control with damping ratio = 1
  tau_nullspace_left << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                       ( (left_arm_data.nullspace_stiffness_.array() * null_space_error.array()).matrix() -
                         (2.0 * left_arm_data.nullspace_stiffness_.array().sqrt() * dq.array()).matrix() );

  //Avoid joint limits
  tau_joint_limit.setZero();

  // (double q_value, double threshold, double magnitude, double upper_bound, double lower_bound) 
  tau_joint_limit(0) = calculateTauJointLimit(q(0), 0.05, 4.0, joint_limits[0][1], joint_limits[0][0]); 
  tau_joint_limit(1) = calculateTauJointLimit(q(1), 0.05, 4.0, joint_limits[1][1], joint_limits[1][0]);
  tau_joint_limit(2) = calculateTauJointLimit(q(2), 0.05, 4.0, joint_limits[2][1], joint_limits[2][0]);
  tau_joint_limit(3) = calculateTauJointLimit(q(3), 0.05, 4.0, joint_limits[3][1], joint_limits[3][0]);
  tau_joint_limit(4) = calculateTauJointLimit(q(4), 0.05, 4.0, joint_limits[4][1], joint_limits[4][0]);
  tau_joint_limit(5) = calculateTauJointLimit(q(5), 0.05, 4.0, joint_limits[5][1], joint_limits[5][0]);
  tau_joint_limit(6) = calculateTauJointLimit(q(6), 0.05, 4.0, joint_limits[6][1], joint_limits[6][0]);



for (int i = 0; i < 7; ++i) {
    tau_joint_limit(i) = std::max(std::min(tau_joint_limit(i), 5.0), -5.0);
}

  tau_relative << jacobian.transpose() * (-left_arm_data.cartesian_stiffness_relative_ * error_relative-
                                      left_arm_data.cartesian_damping_relative_ * (jacobian * dq - jacobian_right * dq_right)); //TODO: MAKE THIS VELOCITY RELATIVE
  // Desired torque
  tau_d_left << tau_task + tau_nullspace_left + coriolis+ tau_joint_limit+ tau_relative ;
  // Saturate torque rate to avoid discontinuities
  tau_d_left << saturateTorqueRateLeft(tau_d_left, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    left_arm_data.joint_handles_[i].setCommand(tau_d_left(i));
  }
}

Eigen::Matrix<double, 7, 1> BiManualCartesianImpedanceControl::saturateTorqueRateLeft(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
    auto& left_arm_data = arms_data_.at(left_arm_id_);
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, left_arm_data.delta_tau_max_),
                                               -left_arm_data.delta_tau_max_);
  }
  return tau_d_saturated;
}

void BiManualCartesianImpedanceControl::updateArmRight() {
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  // get state variables
  franka::RobotState robot_state_right = right_arm_data.state_handle_->getRobotState();

  //JUST FOR DEBUGGING
  ROS_INFO_THROTTLE(1.0, "Current right arm mode: %d", static_cast<int>(robot_state_right.robot_mode));

  Eigen::Vector3d position_d_target;
  Eigen::Quaterniond orientation_d_target;
  Eigen::Vector3d position_d_relative_target;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_target;

  if (controller_state_ == NORMAL_OPERATION) {
      position_d_target = right_arm_data.position_d_;
      orientation_d_target = right_arm_data.orientation_d_;
      position_d_relative_target = right_arm_data.position_d_relative_;
      q_d_nullspace_target = right_arm_data.q_d_nullspace_;
  } else { // COLLISION_DETECTED or RECOVERY_PENDING, use frozen poses
      position_d_target = frozen_pose_right_.position_d;
      orientation_d_target = frozen_pose_right_.orientation_d;
      position_d_relative_target = frozen_pose_right_.position_d_relative;
      q_d_nullspace_target = frozen_pose_right_.q_d_nullspace;
  }


  std::array<double, 49> inertia_array = right_arm_data.model_handle_->getMass();
  std::array<double, 7> coriolis_array = right_arm_data.model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      right_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  std::array<double, 42> jacobian_array_left =
      left_arm_data.model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_left(jacobian_array_left.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state_right.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state_right.dq.data());

  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state_right.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state_right.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  Eigen::MatrixXd jacobian_transpose_pinv;
  franka_bimanual_controllers::pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  franka::RobotState robot_state_left = left_arm_data.state_handle_->getRobotState();
  Eigen::Affine3d transform_left(Eigen::Matrix4d::Map(robot_state_left.O_T_EE.data()));
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq_left(robot_state_left.dq.data());
  Eigen::Vector3d position_left(transform_left.translation());
  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error_right;
  error_right.head(3) << position - position_d_target;

  // calculate the magnitude of the position error
  double position_error_magnitude = error_right.head(3).norm();
  if (position_error_magnitude > delta_lim){
    // scale the position error to the delta_lim
    error_right.head(3) *= (delta_lim / position_error_magnitude);
  }


  geometry_msgs::PoseStamped msg_right;
  msg_right.pose.position.x=position[0];
  msg_right.pose.position.y=position[1];
  msg_right.pose.position.z=position[2];

  msg_right.pose.orientation.x=orientation.x();
  msg_right.pose.orientation.y=orientation.y();
  msg_right.pose.orientation.z=orientation.z();
  msg_right.pose.orientation.w=orientation.w();
  pub_right.publish(msg_right);

  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_ext(robot_state_right.tau_ext_hat_filtered.data());
  Eigen::Matrix<double, 7, 1>  tau_f;

  // Compute the value of the friction
  tau_f(0) =  FI_11/(1+exp(-FI_21*(dq(0)+FI_31))) - TAU_F_CONST_1;
  tau_f(1) =  FI_12/(1+exp(-FI_22*(dq(1)+FI_32))) - TAU_F_CONST_2;
  tau_f(2) =  FI_13/(1+exp(-FI_23*(dq(2)+FI_33))) - TAU_F_CONST_3;
  tau_f(3) =  FI_14/(1+exp(-FI_24*(dq(3)+FI_34))) - TAU_F_CONST_4;
  tau_f(4) =  FI_15/(1+exp(-FI_25*(dq(4)+FI_35))) - TAU_F_CONST_5;
  tau_f(5) =  FI_16/(1+exp(-FI_26*(dq(5)+FI_36))) - TAU_F_CONST_6;
  tau_f(6) =  FI_17/(1+exp(-FI_27*(dq(6)+FI_37))) - TAU_F_CONST_7;

  float iCutOffFrequency=10.0;
  right_arm_data.force_torque+=(-jacobian_transpose_pinv*(tau_ext-tau_f)-right_arm_data.force_torque)*(1-exp(-0.001 * 2.0 * M_PI * iCutOffFrequency));
  geometry_msgs::WrenchStamped force_torque_msg;
  force_torque_msg.wrench.force.x=right_arm_data.force_torque[0];
  force_torque_msg.wrench.force.y=right_arm_data.force_torque[1];
  force_torque_msg.wrench.force.z=right_arm_data.force_torque[2];
  force_torque_msg.wrench.torque.x=right_arm_data.force_torque[3];
  force_torque_msg.wrench.torque.y=right_arm_data.force_torque[4];
  force_torque_msg.wrench.torque.z=right_arm_data.force_torque[5];
  pub_force_torque_right.publish(force_torque_msg);

  Eigen::Matrix<double, 6, 1> error_relative;
  error_relative.head(3) << position - position_left;
  error_relative.tail(3).setZero();
  error_relative.head(3)<< error_relative.head(3) -position_d_relative_target;

  // calculate the magnitude of the relative position error
  double relative_error_magnitude = error_relative.head(3).norm();
  if (relative_error_magnitude > delta_lim){
    // scale the relative position error to the delta_lim
    error_relative.head(3) *= (delta_lim / relative_error_magnitude);
  }
  
  // orientation error
  if (orientation_d_target.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation * orientation_d_target.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  // compute "orientation error"
  error_right.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

  // define orientation error clipping limit
  const double orientation_delta_lim = delta_lim * 3.0;
  // calculate the magnitude of the orientation error
  double orientation_error_magnitude = error_right.tail(3).norm();
  if (orientation_error_magnitude > orientation_delta_lim) {
    // scale the orientation error to the orientation_delta_lim
    error_right.tail(3) *= (orientation_delta_lim / orientation_error_magnitude);
  }

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace_right(7), tau_d(7), tau_joint_limit(7), null_space_error(7), tau_relative(7);

  null_space_error.setZero();
  null_space_error(0)=(q_d_nullspace_target(0) - q(0));
  null_space_error(1)=(q_d_nullspace_target(1) - q(1));
  null_space_error(2)=(q_d_nullspace_target(2) - q(2));
  null_space_error(3)=(q_d_nullspace_target(3) - q(3));
  null_space_error(4)=(q_d_nullspace_target(4) - q(4));
  null_space_error(5)=(q_d_nullspace_target(5) - q(5));
  null_space_error(6)=(q_d_nullspace_target(6) - q(6));
  // Cartesian PD control with damping ratio = 1
  tau_task << jacobian.transpose() * (-right_arm_data.cartesian_stiffness_ * error_right -
                                      right_arm_data.cartesian_damping_ * (jacobian * dq));
  // nullspace PD control with damping ratio = 1
  tau_nullspace_right << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) *
                        ( (right_arm_data.nullspace_stiffness_.array() * null_space_error.array()).matrix() -
                          (2.0 * right_arm_data.nullspace_stiffness_.array().sqrt() * dq.array()).matrix() );

  //Avoid joint limits
  tau_joint_limit.setZero(); // the comment on the right side is the joint limit reported by 
  // (double q_value, double threshold, double magnitude, double upper_bound, double lower_bound) 
  tau_joint_limit(0) = calculateTauJointLimit(q(0), 0.05, 4.0, joint_limits[0][1], joint_limits[0][0]); 
  tau_joint_limit(1) = calculateTauJointLimit(q(1), 0.05, 4.0, joint_limits[1][1], joint_limits[1][0]);
  tau_joint_limit(2) = calculateTauJointLimit(q(2), 0.05, 4.0, joint_limits[2][1], joint_limits[2][0]);
  tau_joint_limit(3) = calculateTauJointLimit(q(3), 0.05, 4.0, joint_limits[3][1], joint_limits[3][0]);
  tau_joint_limit(4) = calculateTauJointLimit(q(4), 0.05, 4.0, joint_limits[4][1], joint_limits[4][0]);
  tau_joint_limit(5) = calculateTauJointLimit(q(5), 0.05, 4.0, joint_limits[5][1], joint_limits[5][0]);
  tau_joint_limit(6) = calculateTauJointLimit(q(6), 0.05, 4.0, joint_limits[6][1], joint_limits[6][0]);



for (int i = 0; i < 7; ++i) {
    tau_joint_limit(i) = std::max(std::min(tau_joint_limit(i), 5.0), -5.0);
}


  tau_relative << jacobian.transpose() * (-right_arm_data.cartesian_stiffness_relative_ * error_relative-
                                      right_arm_data.cartesian_damping_relative_ * (jacobian * dq - jacobian_left * dq_left)); 
  // Desired torque
  tau_d << tau_task + tau_nullspace_right + coriolis+tau_joint_limit+tau_relative;
  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRateRight(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    right_arm_data.joint_handles_[i].setCommand(tau_d(i));
  }
}

Eigen::Matrix<double, 7, 1> BiManualCartesianImpedanceControl::saturateTorqueRateRight(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, right_arm_data.delta_tau_max_),
                                               -right_arm_data.delta_tau_max_);
  }
  return tau_d_saturated;
    }


void BiManualCartesianImpedanceControl::complianceParamCallback(
    franka_combined_bimanual_controllers::dual_arm_compliance_paramConfig& config,
    uint32_t /*level*/) {

   auto& left_arm_data = arms_data_.at(left_arm_id_);
   delta_lim=config.delta_lim; 
   left_arm_data.cartesian_stiffness_.setIdentity();
   left_arm_data.cartesian_stiffness_(0,0)=config.panda_left_translational_stiffness_X;
   left_arm_data.cartesian_stiffness_(1,1)=config.panda_left_translational_stiffness_Y;
   left_arm_data.cartesian_stiffness_(2,2)=config.panda_left_translational_stiffness_Z;
   left_arm_data.cartesian_stiffness_(3,3)=config.panda_left_rotational_stiffness_X;
   left_arm_data.cartesian_stiffness_(4,4)=config.panda_left_rotational_stiffness_Y;
   left_arm_data.cartesian_stiffness_(5,5)=config.panda_left_rotational_stiffness_Z;

  left_arm_data.cartesian_damping_(0,0)=2.0 * sqrt(config.panda_left_translational_stiffness_X)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(1,1)=2.0 * sqrt(config.panda_left_translational_stiffness_Y)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(2,2)=2.0 * sqrt(config.panda_left_translational_stiffness_Z)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(3,3)=2.0 * sqrt(config.panda_left_rotational_stiffness_X)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(4,4)=2.0 * sqrt(config.panda_left_rotational_stiffness_Y)*config.panda_left_damping_ratio;
  left_arm_data.cartesian_damping_(5,5)=2.0 * sqrt(config.panda_left_rotational_stiffness_Z)*config.panda_left_damping_ratio;

  Eigen::AngleAxisd rollAngle_left(config.panda_left_stiffness_roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd yawAngle_left(config.panda_left_stiffness_yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle_left(config.panda_left_stiffness_pitch, Eigen::Vector3d::UnitY());
  Eigen::Quaternion<double> q_left = rollAngle_left *  pitchAngle_left * yawAngle_left;
  Eigen::Matrix3d rotationMatrix_left = q_left.matrix();
  Eigen::Matrix3d rotationMatrix_transpose_left= rotationMatrix_left.transpose();
  left_arm_data.cartesian_stiffness_.topLeftCorner(3, 3) << rotationMatrix_left*left_arm_data.cartesian_stiffness_.topLeftCorner(3, 3)*rotationMatrix_transpose_left;
  left_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3) << rotationMatrix_left*left_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3)*rotationMatrix_transpose_left;


  left_arm_data.nullspace_stiffness_ << config.common_nullspace_stiffness_j1,
                                        config.common_nullspace_stiffness_j2,
                                        config.common_nullspace_stiffness_j3,
                                        config.common_nullspace_stiffness_j4,
                                        config.common_nullspace_stiffness_j5,
                                        config.common_nullspace_stiffness_j6,
                                        config.common_nullspace_stiffness_j7;
  // Update q_d_nullspace_ for the left arm from dynamic reconfigure
  left_arm_data.q_d_nullspace_ << config.panda_left_q_d_nullspace_j1,
                                  config.panda_left_q_d_nullspace_j2,
                                  config.panda_left_q_d_nullspace_j3,
                                  config.panda_left_q_d_nullspace_j4,
                                  config.panda_left_q_d_nullspace_j5,
                                  config.panda_left_q_d_nullspace_j6,
                                  config.panda_left_q_d_nullspace_j7;
   ROS_INFO_STREAM("Left arm nullspace_stiffness: " << left_arm_data.nullspace_stiffness_.transpose());
   ROS_INFO("Left arm q_d_nullspace: %f, %f, %f, %f, %f, %f, %f", left_arm_data.q_d_nullspace_(0), left_arm_data.q_d_nullspace_(1), left_arm_data.q_d_nullspace_(2), left_arm_data.q_d_nullspace_(3), left_arm_data.q_d_nullspace_(4), left_arm_data.q_d_nullspace_(5), left_arm_data.q_d_nullspace_(6));
  left_arm_data.cartesian_stiffness_relative_.setIdentity();
  left_arm_data.cartesian_stiffness_relative_.topLeftCorner(3, 3)
      << config.coupling_translational_stiffness * Eigen::Matrix3d::Identity();
  left_arm_data.cartesian_stiffness_relative_.bottomRightCorner(3, 3)
      << 0.0 * Eigen::Matrix3d::Identity();

  left_arm_data.cartesian_damping_relative_.setIdentity();
  left_arm_data.cartesian_damping_relative_.topLeftCorner(3, 3)
      << 2* sqrt(config.coupling_translational_stiffness) * Eigen::Matrix3d::Identity();
  left_arm_data.cartesian_damping_relative_.bottomRightCorner(3, 3)
          << 0.0 * Eigen::Matrix3d::Identity();
          




  auto& right_arm_data = arms_data_.at(right_arm_id_);

  right_arm_data.cartesian_stiffness_.setIdentity();

  right_arm_data.cartesian_stiffness_(0,0)=config.panda_right_translational_stiffness_X;
  right_arm_data.cartesian_stiffness_(1,1)=config.panda_right_translational_stiffness_Y;
  right_arm_data.cartesian_stiffness_(2,2)=config.panda_right_translational_stiffness_Z;
  right_arm_data.cartesian_stiffness_(3,3)=config.panda_right_rotational_stiffness_X;
  right_arm_data.cartesian_stiffness_(4,4)=config.panda_right_rotational_stiffness_Y;
  right_arm_data.cartesian_stiffness_(5,5)=config.panda_right_rotational_stiffness_Z;

  right_arm_data.cartesian_damping_(0,0)=2.0 * sqrt(config.panda_right_translational_stiffness_X)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(1,1)=2.0 * sqrt(config.panda_right_translational_stiffness_Y)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(2,2)=2.0 * sqrt(config.panda_right_translational_stiffness_Z)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(3,3)=2.0 * sqrt(config.panda_right_rotational_stiffness_X)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(4,4)=2.0 * sqrt(config.panda_right_rotational_stiffness_Y)*config.panda_right_damping_ratio;
  right_arm_data.cartesian_damping_(5,5)=2.0 * sqrt(config.panda_right_rotational_stiffness_Z)*config.panda_right_damping_ratio;

  Eigen::AngleAxisd rollAngle_right(config.panda_right_stiffness_roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd yawAngle_right(config.panda_right_stiffness_yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle_right(config.panda_right_stiffness_pitch, Eigen::Vector3d::UnitY());
  Eigen::Quaternion<double> q_right = rollAngle_right *  pitchAngle_right * yawAngle_right;
  Eigen::Matrix3d rotationMatrix_right = q_right.matrix();
  Eigen::Matrix3d rotationMatrix_transpose_right= rotationMatrix_right.transpose();
  right_arm_data.cartesian_stiffness_.topLeftCorner(3, 3) << rotationMatrix_right*right_arm_data.cartesian_stiffness_.topLeftCorner(3, 3)*rotationMatrix_transpose_right;
  right_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3) << rotationMatrix_right*right_arm_data.cartesian_stiffness_.bottomRightCorner(3, 3)*rotationMatrix_transpose_right;

  right_arm_data.nullspace_stiffness_ << config.common_nullspace_stiffness_j1,
                                         config.common_nullspace_stiffness_j2,
                                         config.common_nullspace_stiffness_j3,
                                         config.common_nullspace_stiffness_j4,
                                         config.common_nullspace_stiffness_j5,
                                         config.common_nullspace_stiffness_j6,
                                         config.common_nullspace_stiffness_j7;
  // Update q_d_nullspace_ for the right arm from dynamic reconfigure
  right_arm_data.q_d_nullspace_ << config.panda_right_q_d_nullspace_j1,
                                   config.panda_right_q_d_nullspace_j2,
                                   config.panda_right_q_d_nullspace_j3,
                                   config.panda_right_q_d_nullspace_j4,
                                   config.panda_right_q_d_nullspace_j5,
                                   config.panda_right_q_d_nullspace_j6,
                                   config.panda_right_q_d_nullspace_j7;
   ROS_INFO_STREAM("Right arm nullspace_stiffness: " << right_arm_data.nullspace_stiffness_.transpose());
   ROS_INFO("Right arm q_d_nullspace: %f, %f, %f, %f, %f, %f, %f", right_arm_data.q_d_nullspace_(0), right_arm_data.q_d_nullspace_(1), right_arm_data.q_d_nullspace_(2), right_arm_data.q_d_nullspace_(3), right_arm_data.q_d_nullspace_(4), right_arm_data.q_d_nullspace_(5), right_arm_data.q_d_nullspace_(6));

  right_arm_data.cartesian_stiffness_relative_.setIdentity();
  right_arm_data.cartesian_stiffness_relative_.topLeftCorner(3, 3)
      << config.coupling_translational_stiffness * Eigen::Matrix3d::Identity();
  right_arm_data.cartesian_stiffness_relative_.bottomRightCorner(3, 3)
      << 0.0 * Eigen::Matrix3d::Identity();

  right_arm_data.cartesian_damping_relative_.setIdentity();
  right_arm_data.cartesian_damping_relative_.topLeftCorner(3, 3)
      << 2* sqrt(config.coupling_translational_stiffness) * Eigen::Matrix3d::Identity();
  right_arm_data.cartesian_damping_relative_.bottomRightCorner(3, 3)
          << 0.0 * Eigen::Matrix3d::Identity();

}

void BiManualCartesianImpedanceControl::freezeDesiredPoses() {
    auto& left_arm_data = arms_data_.at(left_arm_id_);
    auto& right_arm_data = arms_data_.at(right_arm_id_);

    // freeze left arm poses
    frozen_pose_left_.position_d = left_arm_data.position_d_;
    frozen_pose_left_.orientation_d = left_arm_data.orientation_d_;
    frozen_pose_left_.position_d_relative = left_arm_data.position_d_relative_;
    frozen_pose_left_.q_d_nullspace = left_arm_data.q_d_nullspace_;

    // freeze right arm poses
    frozen_pose_right_.position_d = right_arm_data.position_d_;
    frozen_pose_right_.orientation_d = right_arm_data.orientation_d_;
    frozen_pose_right_.position_d_relative = right_arm_data.position_d_relative_;
    frozen_pose_right_.q_d_nullspace = right_arm_data.q_d_nullspace_;

    ROS_WARN("CONTROLLER STATE: COLLISION_DETECTED! Freezing desired poses. Waiting for E-Stop cycle to recover.");
}

double BiManualCartesianImpedanceControl::calculateTauJointLimit(double q_value, double threshold, double magnitude, double upper_bound, double lower_bound) {
    double upper_limit = upper_bound - threshold;
    double lower_limit = lower_bound + threshold;
    if (q_value > (upper_limit)) {
        return -magnitude * (std::exp( std::abs( q_value - upper_limit )/threshold) - 1);
    } else if (q_value < lower_limit) {
        return +magnitude * (std::exp( std::abs( q_value - lower_limit )/threshold) - 1);
    } else {
        return 0;
    }
}

void BiManualCartesianImpedanceControl::equilibriumPoseCallback_left(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  if (controller_state_ != NORMAL_OPERATION) {
    return; // Don't accept new poses while in a non-normal state
  }
  if (!initial_heartbeat_received_.load()) { // atomic read
      return; // Skip update if initial heartbeat not received
  }
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  left_arm_data.position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_(left_arm_data.orientation_d_);
  left_arm_data.orientation_d_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
}

void BiManualCartesianImpedanceControl::equilibriumPoseCallback_right(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  if (controller_state_ != NORMAL_OPERATION) {
    return; // Don't accept new poses while in a non-normal state
  }
  if (!initial_heartbeat_received_.load()) { // atomic read
      return; // Skip update if initial heartbeat not received
  }
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  right_arm_data.position_d_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_(right_arm_data.orientation_d_);
  right_arm_data.orientation_d_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
}


void BiManualCartesianImpedanceControl::equilibriumPoseCallback_relative(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  if (controller_state_ != NORMAL_OPERATION) {
    return; // Don't accept new poses while in a non-normal state
  }
  //This function is receiving the distance from the distance respect the right arm and the left
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  auto&  left_arm_data = arms_data_.at(left_arm_id_);
  left_arm_data.position_d_relative_  << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  right_arm_data.position_d_relative_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  left_arm_data.position_d_relative_=-left_arm_data.position_d_relative_;
}

void BiManualCartesianImpedanceControl::equilibriumConfigurationCallback_right(const sensor_msgs::JointState::ConstPtr& joint) {
  if (controller_state_ != NORMAL_OPERATION) {
    return; // Don't accept new poses while in a non-normal state
  }
  auto& right_arm_data = arms_data_.at(right_arm_id_);
  std::vector<double> read_joint_right;
  read_joint_right= joint -> position;
  right_arm_data.q_d_nullspace_(0) = read_joint_right[0];
  right_arm_data.q_d_nullspace_(1) = read_joint_right[1];
  right_arm_data.q_d_nullspace_(2) = read_joint_right[2];
  right_arm_data.q_d_nullspace_(3) = read_joint_right[3];
  right_arm_data.q_d_nullspace_(4) = read_joint_right[4];
  right_arm_data.q_d_nullspace_(5) = read_joint_right[5];
  right_arm_data.q_d_nullspace_(6) = read_joint_right[6];
}

void BiManualCartesianImpedanceControl::equilibriumConfigurationCallback_left(const sensor_msgs::JointState::ConstPtr& joint) {
  if (controller_state_ != NORMAL_OPERATION) {
    return; // Don't accept new poses while in a non-normal state
  }
  auto& left_arm_data = arms_data_.at(left_arm_id_);
  std::vector<double> read_joint_left;
  read_joint_left= joint -> position;
  left_arm_data.q_d_nullspace_(0) = read_joint_left[0];
  left_arm_data.q_d_nullspace_(1) = read_joint_left[1];
  left_arm_data.q_d_nullspace_(2) = read_joint_left[2];
  left_arm_data.q_d_nullspace_(3) = read_joint_left[3];
  left_arm_data.q_d_nullspace_(4) = read_joint_left[4];
  left_arm_data.q_d_nullspace_(5) = read_joint_left[5];
  left_arm_data.q_d_nullspace_(6) = read_joint_left[6];

}

// Callback function for the set_safety_state service
bool BiManualCartesianImpedanceControl::setSafetyCallback(
    std_srvs::SetBool::Request& req,
    std_srvs::SetBool::Response& res) {
  is_safe_.store(req.data); // atomic write
  res.success = true;
  if (is_safe_.load()) { // atomic read
    res.message = "Controller set to SAFE state.";
    ROS_INFO("Controller set to SAFE state.");
  } else {
    res.message = "Controller set to UNSAFE state.";
    ROS_WARN("Controller set to UNSAFE state.");
  }
  return true;
}

void BiManualCartesianImpedanceControl::heartbeatCallback(const std_msgs::Header::ConstPtr& msg) {
    ros::Time header_stamp = msg->stamp;      // Get the header stamp
    ROS_INFO("Heartbeat received at time %f with timestamp: %f", ros::Time::now().toSec(), header_stamp.toSec());
    {
        // Lock mutex for write
        std::lock_guard<std::mutex> lock(heartbeat_mutex_); 
        // Use the timestamp from the message header as requested
        last_heartbeat_time_ = header_stamp;
        if (!initial_heartbeat_received_.load()) { // atomic read
            initial_heartbeat_received_.store(true); // atomic write
            ROS_INFO("Initial collision detection heartbeat received. Controller operational.");
        }
    }
}
}  // namespace franka_bimanual_controllers

PLUGINLIB_EXPORT_CLASS(franka_bimanual_controllers::BiManualCartesianImpedanceControl,
                       controller_interface::ControllerBase)
