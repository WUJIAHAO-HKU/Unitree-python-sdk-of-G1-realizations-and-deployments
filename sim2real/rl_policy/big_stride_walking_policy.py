"""
BigStrideWalking Policy Deployment for G1 Robot
专门用于部署 BigStrideWalking 策略
"""
import rclpy
from rclpy.node import Node
import numpy as np
import time
import threading
from sshkeyboard import listen_keyboard
import argparse
import yaml
import sys
sys.path.append('./rl_policy')

import onnxruntime
import os
from loguru import logger

from deepmimic_dec_loco import MotionTrackingDecLocoPolicy
from base_policy import BasePolicy

def quat_rotate_inverse_numpy(q, v):
    """将向量从世界坐标系转换到机器人局部坐标系"""
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    
    return a - b + c

class BigStrideWalkingPolicy(MotionTrackingDecLocoPolicy):
    """BigStrideWalking 策略部署类，使用 short_history 和 gait_phase"""
    
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit=False,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4):
        """
        初始化 BigStrideWalking 策略
        
        Args:
            config: 配置文件字典（需要包含 short_history 配置）
            node: ROS2节点
            model_path: ONNX模型路径
            use_jit: 是否使用JIT模型
            rl_rate: 控制频率 (Hz)
            policy_action_scale: 动作缩放因子
            decimation: 控制降采样倍数
        """
        # 使用 short_history 而不是 history_loco
        config["USE_HISTORY_LOCO"] = True
        config["USE_HISTORY_MIMIC"] = False
        
        # 设置 short_history 配置（从训练配置中读取）
        if "short_history_config" not in config:
            # 默认的 short_history 配置（匹配 BigStrideWalking 训练配置）
            config["short_history_config"] = {
                "base_ang_vel": 4,
                "projected_gravity": 4,
                "dof_pos": 4,
                "dof_vel": 4,
                "actions": 4,
                "gait_phase": 4,
            }
        
        # 将 short_history_config 添加到 history_config，以便历史处理器初始化
        # 同时需要设置对应的 obs_dims
        if "history_config" not in config:
            config["history_config"] = {}
        config["history_config"].update(config["short_history_config"])
        
        # 确保 obs_dims 包含所有需要的键
        if "obs_dims" not in config:
            config["obs_dims"] = {}
        # 为 short_history 中的键设置 obs_dims（如果不存在）
        for key in config["short_history_config"].keys():
            if key not in config["obs_dims"]:
                if key == "gait_phase":
                    config["obs_dims"][key] = 2  # sin/cos
                elif key in ["base_ang_vel", "projected_gravity"]:
                    config["obs_dims"][key] = 3
                else:
                    # dof_pos, dof_vel, actions 使用 num_dofs（稍后会在父类中设置）
                    config["obs_dims"][key] = 29  # 临时值，会被覆盖
        
        # MotionTrackingDecLocoPolicy 需要 loco_model_path 和 mimic_model_paths
        # 对于 BigStrideWalking，我们只需要 locomotion 策略，不需要 mimic
        # 设置一个空的 mimic_models 配置，避免 setup_mimic_policies 报错
        if "mimic_models" not in config:
            config["mimic_models"] = {}
        
        super().__init__(config, 
                         node, 
                         loco_model_path=model_path, 
                         mimic_model_paths=None,
                         use_jit=use_jit,
                         rl_rate=rl_rate, 
                         policy_action_scale=policy_action_scale, 
                         decimation=decimation)
        
        # 获取训练时的 DOF 数量（用于观测构建）
        self.training_dof_size = config.get("TRAINING_DOF_SIZE", self.num_dofs)
        
        # 更新 obs_dims 中的 dof 相关键（使用训练时的 DOF 数量，而不是实际的 num_dofs）
        # BigStrideWalking 训练时使用 23 自由度，但部署时物理系统是 29 自由度
        for key in ["dof_pos", "dof_vel", "actions"]:
            if key in config["obs_dims"]:
                config["obs_dims"][key] = self.training_dof_size
        
        # 重新初始化历史处理器（使用更新后的 obs_dims）
        if self.use_history:
            from sim2real.utils.history_handler import HistoryHandler
            self.history_handler = HistoryHandler(config["history_config"], config["obs_dims"])
        
        # BigStrideWalking 特定参数
        self.gait_period = config.get("GAIT_PERIOD", 0.9)
        self.phase_time = np.zeros((1, 1))
        self.frame_idx = 0
        start_time = self.node.get_clock().now().nanoseconds / 1e9
        self.frame_start_time = start_time
        
        logger.info(f"BigStrideWalking Policy initialized:")
        logger.info(f"  Gait period: {self.gait_period}s")
        logger.info(f"  Control rate: {rl_rate}Hz")
        logger.info(f"  Action scale: {policy_action_scale}")
        logger.info(f"  Training DOF size: {self.training_dof_size} (actual robot DOF: {self.num_dofs})")
        logger.info(f"  Training DOF size: {self.training_dof_size} (actual: {self.num_dofs})")
    
    def setup_policy(self, model_path, use_jit):
        """
        重写 setup_policy，只加载 locomotion 策略，不加载 mimic 策略
        """
        if not use_jit:
            self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
            def policy_act(obs):
                return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        else:
            raise NotImplementedError("JIT not implemented yet.")
        self.policy_locomotion = policy_act
        # 不调用 setup_mimic_policies，因为 BigStrideWalking 不需要 mimic 策略
        # Default policy is locomotion
        self.policy = self.policy_locomotion
    
    
    def _get_obs_phase_time(self):
        """获取步态相位时间（0-1）"""
        cur_time = self.node.get_clock().now().nanoseconds / 1e9
        phase_time = (cur_time - self.frame_start_time) % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time
    
    def _get_obs_short_history(self, obs_dims={}):
        """获取 short_history（匹配 BigStrideWalking 训练配置）"""
        assert "short_history_config" in self.config.keys()
        history_config = self.config["short_history_config"]
        history_list = []
        
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            obs_dim = obs_dims.get(key, history_array.shape[2])
            history_array = history_array[:, :, :obs_dim]
            history_array = history_array.reshape(history_array.shape[0], -1)
            history_list.append(history_array)
        
        return np.concatenate(history_list, axis=1)
    
    def prepare_obs_for_rl(self, robot_state_data):
        """
        准备RL观测向量（385维，匹配 BigStrideWalking 训练配置）
        
        观测顺序:
        - base_ang_vel (3)
        - projected_gravity (3)
        - dof_pos (23)
        - dof_vel (23)
        - actions (23)
        - gait_phase (2: sin/cos)
        - short_history (308: 各组件历史 × 4)
        """
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
        
        dof_pos_minus_default = dof_pos - self.default_dof_angles
        
        # 只取前 training_dof_size 个 DOF（匹配训练时的配置）
        dof_pos_minus_default = dof_pos_minus_default[:, :self.training_dof_size]
        dof_vel = dof_vel[:, :self.training_dof_size]
        
        # 确保 last_policy_action 只包含前 training_dof_size 个元素
        if self.last_policy_action.shape[1] > self.training_dof_size:
            last_action_for_obs = self.last_policy_action[:, :self.training_dof_size]
        else:
            last_action_for_obs = self.last_policy_action
        
        # 投影重力
        v = np.array([[0, 0, -1]])
        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        # 步态相位（sin/cos）
        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2*np.pi*phase_time)
        cos_phase = np.cos(2*np.pi*phase_time)
        gait_phase = np.concatenate([sin_phase, cos_phase], axis=1)
        
        # 获取 short_history（使用训练时的 DOF 数量）
        obs_dims = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "dof_pos": self.training_dof_size,
            "dof_vel": self.training_dof_size,
            "actions": self.training_dof_size,
            "gait_phase": 2,
        }
        short_history = self._get_obs_short_history(obs_dims)
        short_history *= self.obs_scales.get("short_history", 1.0)
        
        # 构建观测向量（按照训练时的顺序）
        obs = np.concatenate([
            base_ang_vel * self.obs_scales.get("base_ang_vel", 0.25),
            projected_gravity * self.obs_scales.get("projected_gravity", 1.0),
            dof_pos_minus_default * self.obs_scales.get("dof_pos", 1.0),
            dof_vel * self.obs_scales.get("dof_vel", 0.05),
            last_action_for_obs * self.obs_scales.get("actions", 1.0),
            gait_phase * self.obs_scales.get("gait_phase", 1.0),
            short_history,
        ], axis=1).astype(np.float32)
        
        # 更新历史
        if self.history_handler:
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales.get("base_ang_vel", 0.25))
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales.get("projected_gravity", 1.0))
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales.get("dof_pos", 1.0))
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales.get("dof_vel", 0.05))
            self.history_handler.add("actions", last_action_for_obs*self.obs_scales.get("actions", 1.0))
            self.history_handler.add("gait_phase", gait_phase*self.obs_scales.get("gait_phase", 1.0))
        
        return obs
    
    def get_policy_action(self, robot_state_data):
        """
        重写 get_policy_action，确保 last_policy_action 使用正确的维度
        """
        # 准备观测
        obs = self.prepare_obs_for_rl(robot_state_data)
        
        # 策略推理
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # BigStrideWalking 策略输出 training_dof_size 维动作
        action_dim = policy_action.shape[1]
        if action_dim >= self.training_dof_size:
            lower_body_action = policy_action[:, :self.training_dof_size]
        else:
            lower_body_action = np.concatenate(
                [policy_action, np.zeros((1, self.training_dof_size - action_dim))],
                axis=1
            )
        
        # 更新 last_policy_action（仅前 training_dof_size 维度）
        self.last_policy_action[:, :self.training_dof_size] = lower_body_action.copy()
        if self.last_policy_action.shape[1] > self.training_dof_size:
            self.last_policy_action[:, self.training_dof_size:] = 0.0
        
        # 缩放动作并恢复成 num_dofs 维（补零）
        scaled_policy_action = lower_body_action * self.policy_action_scale
        if self.num_dofs > self.training_dof_size:
            scaled_policy_action = np.concatenate(
                [scaled_policy_action, np.zeros((1, self.num_dofs - self.training_dof_size))],
                axis=1
            )
        
        return scaled_policy_action

    def handle_keyboard_button(self, keycode):
        """
        BigStrideWalking 不支持 mimic 切换，复用 BasePolicy 的按键逻辑
        """
        BasePolicy.handle_keyboard_button(self, keycode)
        if keycode in ["[", "]", ";", "'"]:
            self.node.get_logger().info("BigStrideWalking policy does not support mimic switching keys.")

    def start_key_listener(self):
        """
        重写键盘监听，使用 sshkeyboard 的阻塞式监听
        """
        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass
        listen_keyboard(on_press=on_press)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BigStrideWalking Policy Deployment for G1')
    parser.add_argument('--config', type=str, default='config/g1_29dof_hist.yaml',
                       help='Config file path')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--training_config', type=str, default=None,
                       help='Training config file path (for loading obs_scales and other params)')
    parser.add_argument('--use_jit', action='store_true', default=False,
                       help='Use JIT model')
    parser.add_argument('--rl_rate', type=int, default=50,
                       help='Control frequency in Hz (default: 50)')
    parser.add_argument('--action_scale', type=float, default=0.25,
                       help='Action scaling factor (default: 0.25)')
    parser.add_argument('--gait_period', type=float, default=0.9,
                       help='Gait period in seconds (default: 0.9)')
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # 如果提供了训练配置，从中加载参数
    training_dof_size = 29  # 默认值
    if args.training_config:
        with open(args.training_config) as file:
            training_config = yaml.load(file, Loader=yaml.FullLoader)
        
        # 从训练配置中读取 DOF 数量
        robot_config = training_config.get("robot", {})
        training_dof_size = robot_config.get("dof_obs_size", 29)
        dof_names = robot_config.get("dof_names", [])
        if dof_names:
            training_dof_size = len(dof_names)
        
        # 加载观测缩放
        obs_config = training_config.get("obs", {})
        obs_scales = obs_config.get("obs_scales", {})
        config["obs_scales"].update(obs_scales)
        
        # 加载 short_history 配置
        obs_auxiliary = obs_config.get("obs_auxiliary", {})
        if "short_history" in obs_auxiliary:
            config["short_history_config"] = obs_auxiliary["short_history"]
        
        # 加载步态周期
        env_config = training_config.get("env", {}).get("config", {})
        if "gait_period" in env_config:
            config["GAIT_PERIOD"] = env_config["gait_period"]
    
    # 保存训练时的 DOF 数量，用于观测构建
    config["TRAINING_DOF_SIZE"] = training_dof_size
    
    # 命令行参数覆盖
    if args.gait_period:
        config["GAIT_PERIOD"] = args.gait_period
    
    # 初始化ROS2
    rclpy.init(args=None)
    node = rclpy.create_node('big_stride_walking_policy_node')
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    
    # 创建并运行策略
    logger.info(f"Loading BigStrideWalking policy from: {args.model_path}")
    policy = BigStrideWalkingPolicy(
        config=config,
        node=node,
        model_path=args.model_path,
        use_jit=args.use_jit,
        rl_rate=args.rl_rate,
        policy_action_scale=args.action_scale
    )
    
    logger.info("BigStrideWalking policy ready. Press ']' to activate, 'i' for initial position, 'o' for emergency stop")
    policy.run()
    rclpy.shutdown()

