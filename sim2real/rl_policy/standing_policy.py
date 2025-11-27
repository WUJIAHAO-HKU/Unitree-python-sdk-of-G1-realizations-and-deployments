"""
Standing Policy Deployment for G1 Robot
使用训练好的站立策略进行真机部署
"""
import rclpy
from rclpy.node import Node
import numpy as np
import time
import pygame
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import threading
from sshkeyboard import listen_keyboard
import argparse
import yaml
import sys
sys.path.append('./rl_policy')

import onnxruntime
import os
from loguru import logger

from base_policy import BasePolicy

def quat_rotate_inverse_numpy(q, v):
    """将向量从世界坐标系转换到机器人局部坐标系"""
    shape = q.shape
    q_w = q[:, 0]  # 四元数标量部分
    q_vec = q[:, 1:]  # 四元数向量部分
    
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    
    return a - b + c

class StandingPolicy(BasePolicy):
    """站立策略部署类"""
    
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit=False,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        """
        初始化站立策略
        
        Args:
            config: 配置文件字典
            node: ROS2节点
            model_path: ONNX模型路径
            use_jit: 是否使用JIT模型（暂不支持）
            rl_rate: 控制频率 (Hz)
            policy_action_scale: 动作缩放因子
            decimation: 控制降采样倍数
        """
        # 站立任务不需要历史缓冲
        config["USE_HISTORY"] = False
        
        super().__init__(config, node, model_path, use_jit, rl_rate, policy_action_scale, decimation)
        
        # 站立任务特定参数
        self.target_base_height = config.get("target_base_height", 0.78)
        self.desired_feet_distance = config.get("target_feet_distance", 0.3)
        
        # 观测缩放（从训练配置中读取）
        self.obs_scales = config.get("obs_scales", {
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "actions": 1.0,
            "base_lin_vel": 2.0,
            "base_ang_vel": 0.25,
            "projected_gravity": 1.0,
            "standing_target_height": 1.0,
            "feet_distance": 1.0,
        })
        
        logger.info(f"Standing Policy initialized:")
        logger.info(f"  Target base height: {self.target_base_height}m")
        logger.info(f"  Desired feet distance: {self.desired_feet_distance}m")
        logger.info(f"  Control rate: {rl_rate}Hz")
        logger.info(f"  Action scale: {policy_action_scale}")
    
    def prepare_obs_for_rl(self, robot_state_data):
        """
        准备RL观测向量（80维）
        
        观测顺序（与训练时一致）:
        - dof_pos (23)
        - dof_vel (23)
        - actions (23)
        - base_lin_vel (3)
        - base_ang_vel (3)
        - projected_gravity (3)
        - standing_target_height (1)
        - feet_distance (1)
        """
        # robot_state_data格式:
        # [:3]: base pos
        # [3:7]: base quaternion (w, x, y, z)
        # [7:7+dof_num]: joint angles
        # [7+dof_num: 7+dof_num+3]: base linear velocity
        # [7+dof_num+3: 7+dof_num+6]: base angular velocity
        # [7+dof_num+6: 7+dof_num+6+dof_num]: joint velocities
        
        base_quat = robot_state_data[:, 3:7]
        base_lin_vel = robot_state_data[:, 7+self.num_dofs:7+self.num_dofs+3]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
        
        # 计算投影重力（从世界坐标系到机器人局部坐标系）
        v = np.array([[0, 0, -1]])  # 重力方向（世界坐标系）
        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        # 站立目标高度（常数）
        standing_target_height = np.array([[self.target_base_height]])
        
        # 计算双脚距离（简化：使用默认值，真机上需要从传感器获取）
        # TODO: 从真实传感器获取双脚位置
        feet_distance = np.array([[self.desired_feet_distance]])
        
        # 构建观测向量（按照训练时的顺序）
        obs_components = [
            dof_pos * self.obs_scales.get("dof_pos", 1.0),
            dof_vel * self.obs_scales.get("dof_vel", 0.05),
            self.last_policy_action * self.obs_scales.get("actions", 1.0),
            base_lin_vel * self.obs_scales.get("base_lin_vel", 2.0),
            base_ang_vel * self.obs_scales.get("base_ang_vel", 0.25),
            projected_gravity * self.obs_scales.get("projected_gravity", 1.0),
            standing_target_height * self.obs_scales.get("standing_target_height", 1.0),
            feet_distance * self.obs_scales.get("feet_distance", 1.0),
        ]
        
        obs = np.concatenate(obs_components, axis=1).astype(np.float32)
        
        # 验证观测维度
        expected_dim = 23 + 23 + 23 + 3 + 3 + 3 + 1 + 1  # 80
        if obs.shape[1] != expected_dim:
            logger.warning(f"Observation dimension mismatch: got {obs.shape[1]}, expected {expected_dim}")
        
        return obs
    
    def start_key_listener(self):
        """键盘监听（继承自BasePolicy）"""
        def on_press(key):
            if key == 'i':
                logger.info("Setting robot to initial position")
                self.get_ready_state = True
                self.init_count = 0
            elif key == 'o':
                logger.warning("EMERGENCY STOP!")
                self.use_policy_action = False
            elif key == 'p':
                logger.info("Activating policy")
                self.use_policy_action = True
        
        listen_keyboard(on_press=on_press)

def load_training_config(training_config_path):
    """
    从训练配置文件中加载配置，并转换为部署格式
    
    Args:
        training_config_path: 训练配置文件的路径
        
    Returns:
        转换后的配置字典
    """
    with open(training_config_path) as file:
        training_config = yaml.load(file, Loader=yaml.FullLoader)
    
    # 加载部署基础配置（用于获取DDS、ROS等通用设置）
    base_config_path = os.path.join(os.path.dirname(__file__), '../config/g1_29dof_hist.yaml')
    with open(base_config_path) as file:
        deploy_config = yaml.load(file, Loader=yaml.FullLoader)
    
    # 从训练配置中提取机器人配置
    robot_config = training_config.get("robot", {})
    dof_names = robot_config.get("dof_names", [])
    num_dofs = len(dof_names)
    
    # 提取默认关节角度（按dof_names顺序）
    default_joint_angles_dict = robot_config.get("init_state", {}).get("default_joint_angles", {})
    default_dof_angles = [default_joint_angles_dict.get(name, 0.0) for name in dof_names]
    
    # 提取PD增益（按关节类型）
    control_config = robot_config.get("control", {})
    stiffness_dict = control_config.get("stiffness", {})
    damping_dict = control_config.get("damping", {})
    
    # 映射关节名称到关节类型
    def get_joint_type(joint_name):
        if "hip_yaw" in joint_name:
            return "hip_yaw"
        elif "hip_roll" in joint_name:
            return "hip_roll"
        elif "hip_pitch" in joint_name:
            return "hip_pitch"
        elif "knee" in joint_name:
            return "knee"
        elif "ankle_pitch" in joint_name:
            return "ankle_pitch"
        elif "ankle_roll" in joint_name:
            return "ankle_roll"
        elif "waist_yaw" in joint_name:
            return "waist_yaw"
        elif "waist_roll" in joint_name:
            return "waist_roll"
        elif "waist_pitch" in joint_name:
            return "waist_pitch"
        elif "shoulder_pitch" in joint_name:
            return "shoulder_pitch"
        elif "shoulder_roll" in joint_name:
            return "shoulder_roll"
        elif "shoulder_yaw" in joint_name:
            return "shoulder_yaw"
        elif "elbow" in joint_name:
            return "elbow"
        else:
            return "unknown"
    
    # 构建23自由度的KP和KD列表（只包含控制的关节）
    motor_kp = [stiffness_dict.get(get_joint_type(name), 100.0) for name in dof_names]
    motor_kd = [damping_dict.get(get_joint_type(name), 2.5) for name in dof_names]
    
    # 构建WeakMotorJointIndex（23自由度版本，只包含控制的关节）
    weak_motor_joint_index = {}
    for i, name in enumerate(dof_names):
        weak_motor_joint_index[name] = i
    
    # 更新部署配置
    deploy_config["NUM_JOINTS"] = num_dofs  # 23个控制的关节
    deploy_config["NUM_MOTORS"] = 29  # 物理电机数量仍然是29
    
    # DEFAULT_DOF_ANGLES: 23个关节的默认角度（用于策略）
    deploy_config["DEFAULT_DOF_ANGLES"] = default_dof_angles
    
    # DEFAULT_MOTOR_ANGLES: 29个电机的默认角度（保持原值，包括手腕）
    # 前23个电机使用策略控制的默认角度，后6个手腕电机保持原配置的默认角度
    base_default_motor_angles = deploy_config["DEFAULT_MOTOR_ANGLES"]
    deploy_config["DEFAULT_MOTOR_ANGLES"] = default_dof_angles + base_default_motor_angles[num_dofs:]
    
    # MOTOR_KP 和 MOTOR_KD: 前23个使用训练配置的值，后6个手腕使用原配置的值
    base_motor_kp = deploy_config["MOTOR_KP"]
    base_motor_kd = deploy_config["MOTOR_KD"]
    deploy_config["MOTOR_KP"] = motor_kp + base_motor_kp[num_dofs:]
    deploy_config["MOTOR_KD"] = motor_kd + base_motor_kd[num_dofs:]
    
    # MOTOR2JOINT: 29个电机映射到23个关节（或-1表示使用默认值）
    # 前23个电机（索引0-22）映射到对应的关节（索引0-22）
    # 后6个电机（手腕，索引23-28）映射为-1，表示不受策略控制，使用默认值
    deploy_config["MOTOR2JOINT"] = list(range(num_dofs)) + [-1] * (29 - num_dofs)
    
    # JOINT2MOTOR: 23个关节映射到29个电机
    # 前23个关节（索引0-22）直接映射到前23个电机（索引0-22）
    # 注意：JOINT2MOTOR的长度应该是NUM_JOINTS（23），但send_command中实际使用的是MOTOR2JOINT
    # 为了兼容性，我们保持JOINT2MOTOR的长度为29，但只使用前23个
    deploy_config["JOINT2MOTOR"] = list(range(num_dofs)) + list(range(num_dofs, 29))
    
    # 关节位置限制：只取前23个关节的限制值
    base_motor_pos_lower = deploy_config.get("motor_pos_lower_limit_list", [])
    base_motor_pos_upper = deploy_config.get("motor_pos_upper_limit_list", [])
    if len(base_motor_pos_lower) >= num_dofs:
        deploy_config["motor_pos_lower_limit_list"] = base_motor_pos_lower[:num_dofs]
    if len(base_motor_pos_upper) >= num_dofs:
        deploy_config["motor_pos_upper_limit_list"] = base_motor_pos_upper[:num_dofs]
    
    # 关节速度限制：只取前23个关节的限制值
    base_motor_vel_limit = deploy_config.get("motor_vel_limit_list", [])
    if len(base_motor_vel_limit) >= num_dofs:
        deploy_config["motor_vel_limit_list"] = base_motor_vel_limit[:num_dofs]
    
    # 关节力矩限制：只取前23个关节的限制值
    base_motor_effort_limit = deploy_config.get("motor_effort_limit_list", [])
    if len(base_motor_effort_limit) >= num_dofs:
        deploy_config["motor_effort_limit_list"] = base_motor_effort_limit[:num_dofs]
    
    deploy_config["WeakMotorJointIndex"] = weak_motor_joint_index
    
    # 提取观测缩放
    obs_config = training_config.get("obs", {})
    obs_scales = obs_config.get("obs_scales", {})
    deploy_config["obs_scales"] = obs_scales
    
    # 提取动作缩放
    action_scale = control_config.get("action_scale", 0.25)
    deploy_config["action_scale"] = action_scale
    
    # 提取目标高度和双脚距离
    env_config = training_config.get("env", {}).get("config", {})
    deploy_config["target_base_height"] = env_config.get("target_base_height", 0.78)
    deploy_config["target_feet_distance"] = env_config.get("target_feet_distance", 0.3)
    
    logger.info(f"Loaded training config: {num_dofs} DOFs")
    logger.info(f"  DOF names: {dof_names}")
    logger.info(f"  Action scale: {action_scale}")
    logger.info(f"  Target height: {deploy_config['target_base_height']}m")
    
    return deploy_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standing Policy Deployment for G1')
    parser.add_argument('--config', type=str, default=None,
                       help='Training config file path (if not provided, will use --training_config)')
    parser.add_argument('--training_config', type=str, 
                       default='logs/Standing/20251017_162813-My_Standing_Task-standing-g1_29dof_anneal_23dof/config.yaml',
                       help='Training config file path (overrides --config)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--use_jit', action='store_true', default=False,
                       help='Use JIT model (not supported for standing policy)')
    parser.add_argument('--rl_rate', type=int, default=50,
                       help='Control frequency in Hz (default: 50)')
    parser.add_argument('--action_scale', type=float, default=None,
                       help='Action scaling factor (default: from training config)')
    parser.add_argument('--target_height', type=float, default=None,
                       help='Target base height in meters (default: from training config)')
    parser.add_argument('--feet_distance', type=float, default=None,
                       help='Desired feet distance in meters (default: from training config)')
    args = parser.parse_args()
    
    # 加载训练配置并转换为部署格式
    training_config_path = args.training_config if args.training_config else args.config
    if not training_config_path:
        raise ValueError("Must provide either --config or --training_config")
    
    config = load_training_config(training_config_path)
    
    # 命令行参数覆盖配置
    if args.action_scale is not None:
        config["action_scale"] = args.action_scale
    if args.target_height is not None:
        config["target_base_height"] = args.target_height
    if args.feet_distance is not None:
        config["target_feet_distance"] = args.feet_distance
    
    # 初始化ROS2
    rclpy.init(args=None)
    node = rclpy.create_node('standing_policy_node')
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    
    # 创建并运行策略
    logger.info(f"Loading standing policy from: {args.model_path}")
    # 使用配置中的action_scale，如果命令行提供了则覆盖
    action_scale = args.action_scale if args.action_scale is not None else config.get("action_scale", 0.25)
    standing_policy = StandingPolicy(
        config=config,
        node=node,
        model_path=args.model_path,
        use_jit=args.use_jit,
        rl_rate=args.rl_rate,
        policy_action_scale=action_scale
    )
    
    logger.info("Standing policy ready. Press 'p' to activate, 'i' for initial position, 'o' for emergency stop")
    standing_policy.run()
    rclpy.shutdown()
