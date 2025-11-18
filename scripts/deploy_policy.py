"""
G1机器人策略部署脚本
将训练好的强化学习策略部署到真实G1机器人上
"""
import time
import sys
import os
import numpy as np
import torch
from pathlib import Path

# 添加humanoidverse路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# 导入状态读取器
from get_robot_state import RobotStateReader, JOINT_NAMES

# G1机器人关节索引（23自由度）
G1_NUM_MOTOR = 29  # 总共有29个电机，但只使用23个

class G1JointIndex:
    # 左腿 (6个关节)
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    
    # 右腿 (6个关节)
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    
    # 腰部 (3个关节)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    
    # 左臂 (4个关节)
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    
    # 右臂 (4个关节)
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25

class PolicyDeployer:
    def __init__(self, checkpoint_path, network_interface=None):
        """
        初始化策略部署器
        
        Args:
            checkpoint_path: 训练好的模型checkpoint路径 (.pt文件)
            network_interface: 网络接口名称（如 'wlp0s20f3'）
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.network_interface = network_interface
        self.device = torch.device("cpu")  # 部署时使用CPU
        
        # 控制参数
        self.control_dt = 0.01  # 控制频率 100Hz
        self.action_scale = 0.25  # 动作缩放（与训练时一致）
        
        # 需要置0的自由度索引（23维动作空间中的索引，0-22）
        # 根据g1_29dof_anneal_23dof.yaml中的dof_names顺序：
        # 0-5: 左腿, 6-11: 右腿, 12: waist_yaw, 13: waist_roll, 14: waist_pitch
        # 15: left_shoulder_pitch, 16: left_shoulder_roll, 17: left_shoulder_yaw, 18: left_elbow
        # 19: right_shoulder_pitch, 20: right_shoulder_roll, 21: right_shoulder_yaw, 22: right_elbow
        # 置0：腰部2个(13,14) + 左臂2个(16,17) + 右臂2个(20,21)
        self.zero_action_indices = [13, 14, 16, 17, 20, 21]
        
        # PD控制参数（与训练时一致）
        self.kp = np.array([
            100, 100, 100, 200, 20, 20,      # 左腿 (0-5)
            100, 100, 100, 200, 20, 20,      # 右腿 (6-11)
            400, 400, 400,                   # 腰部 (12-14): waist_yaw, waist_roll, waist_pitch
            90, 60, 20, 60,                  # 左臂 (15-18): shoulder_pitch, roll, yaw, elbow
            90, 60, 20, 60                   # 右臂 (19-22): shoulder_pitch, roll, yaw, elbow
        ])
        
        self.kd = np.array([
            2.5, 2.5, 2.5, 5.0, 0.2, 0.1,    # 左腿
            2.5, 2.5, 2.5, 5.0, 0.2, 0.1,    # 右腿
            5.0, 5.0, 5.0,                   # 腰部
            2.0, 1.0, 0.4, 1.0,              # 左臂
            2.0, 1.0, 0.4, 1.0               # 右臂
        ])
        
        # 状态和历史
        self.state_reader = None
        self.actor_model = None
        self.low_cmd = None
        self.lowcmd_publisher = None
        self.crc = CRC()
        
        # 观测历史（用于short_history）
        self.history_length = 4
        self.obs_history = {
            'base_ang_vel': [],
            'projected_gravity': [],
            'dof_pos': [],
            'dof_vel': [],
            'actions': [],
            'gait_phase': []
        }
        self.last_actions = np.zeros(23)
        
        # 步态相位
        self.gait_phase = 0.0
        self.phase_increment = 2.0 * np.pi * 0.667 * self.control_dt  # 目标步频0.667Hz
        
        # 打印置0的自由度信息
        print(f"\n📌 自由度配置:")
        print(f"   总自由度: 23维")
        print(f"   置0的自由度: {len(self.zero_action_indices)}个")
        print(f"   实际控制: {23 - len(self.zero_action_indices)}个自由度")
        print(f"   置0索引: {self.zero_action_indices}")
        print(f"   - 腰部: waist_roll(13), waist_pitch(14)")
        print(f"   - 左臂: shoulder_roll(16), shoulder_yaw(17)")
        print(f"   - 右臂: shoulder_roll(20), shoulder_yaw(21)")
        print()
        
    def load_model(self):
        """加载训练好的策略模型"""
        print(f"📦 加载模型: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.checkpoint_path}")
        
        # 尝试从checkpoint目录加载config
        config_path = self.checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = self.checkpoint_path.parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"未找到配置文件: {config_path}\n"
                "请确保checkpoint目录中包含config.yaml文件"
            )
        
        # 加载配置
        from omegaconf import OmegaConf
        from humanoidverse.utils.config_utils import pre_process_config
        
        print(f"📋 加载配置: {config_path}")
        config = OmegaConf.load(config_path)
        pre_process_config(config)
        
        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'actor_model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint中未找到 'actor_model_state_dict'")
        
        # 构建模型（需要环境来获取观测维度）
        # 由于我们无法在部署时创建完整环境，这里提供一个简化方案
        # 方案1: 使用导出的ONNX模型（推荐）
        onnx_path = self.checkpoint_path.parent / "exported" / (self.checkpoint_path.stem + ".onnx")
        if onnx_path.exists():
            print(f"✅ 找到ONNX模型: {onnx_path}")
            print("   请使用ONNX运行时加载模型")
            print("   或使用 humanoidverse/eval_agent.py 导出ONNX模型")
            raise NotImplementedError(
                "ONNX模型加载需要额外实现。\n"
                "建议：先运行 eval_agent.py 导出ONNX模型，然后使用ONNX Runtime加载。"
            )
        
        # 方案2: 从checkpoint重建模型（需要完整配置）
        print("⚠️  需要从checkpoint重建模型")
        print("   这需要完整的训练配置和环境")
        print("\n建议的解决方案:")
        print("   1. 使用 humanoidverse/eval_agent.py 导出ONNX模型")
        print("   2. 或创建一个简化的模型加载脚本")
        print("   3. 或直接使用 humanoidverse/eval_agent.py 进行部署测试")
        
        # 这里提供一个框架，需要根据实际情况完善
        from humanoidverse.agents.modules.ppo_modules import PPOActor
        
        # 获取观测维度（从配置中）
        algo_obs_dim_dict = config.robot.algo_obs_dim_dict
        num_actions = config.robot.actions_dim
        
        # 创建actor模型
        self.actor_model = PPOActor(
            obs_dim_dict=algo_obs_dim_dict,
            module_config_dict=config.algo.config.module_dict.actor,
            num_actions=num_actions,
            init_noise_std=config.algo.config.init_noise_std
        ).to(self.device)
        
        # 加载权重
        self.actor_model.load_state_dict(checkpoint['actor_model_state_dict'])
        self.actor_model.eval()
        
        print("✅ 模型加载成功")
    
    def init_robot_connection(self):
        """初始化机器人连接"""
        print("🔌 初始化机器人连接...")
        
        # 初始化通道
        if self.network_interface:
            ChannelFactoryInitialize(0, self.network_interface)
        else:
            ChannelFactoryInitialize(0)
        
        # 创建状态读取器
        self.state_reader = RobotStateReader()
        self.state_reader.Init(self.network_interface)
        
        # 等待收到第一个状态
        print("⏳ 等待机器人状态...")
        timeout = 10.0
        start_time = time.time()
        while not self.state_reader.state_received:
            if time.time() - start_time > timeout:
                raise TimeoutError("超时：未能收到机器人状态")
            time.sleep(0.1)
        print("✅ 已收到机器人状态")
        
        # 初始化运动切换器（释放高级控制模式）
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        
        status, result = msc.CheckMode()
        while result['name']:
            print(f"🔄 释放控制模式: {result['name']}")
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)
        print("✅ 已切换到低级控制模式")
        
        # 创建控制命令发布者
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        # 初始化控制命令
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._init_low_cmd()
        
    def _init_low_cmd(self):
        """初始化低级控制命令"""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        self.low_cmd.mode_pr = 0  # PR模式
        
        # 初始化所有电机
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1  # 启用
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
    
    def build_observation(self):
        """
        构建观测向量
        根据 big_stride_obs_basic.yaml 的配置构建观测
        """
        if not self.state_reader.state_received:
            return None
        
        # 获取基础状态
        joint_pos = self.state_reader.get_joint_positions()  # 23维
        joint_vel = self.state_reader.get_joint_velocities()  # 23维
        imu = self.state_reader.get_imu_data()
        
        # 1. base_ang_vel (3维) - 角速度（机器人坐标系）
        base_ang_vel = imu['gyroscope']
        
        # 2. projected_gravity (3维) - 投影重力
        # 简化：使用RPY角度计算投影重力
        rpy = imu['rpy']
        projected_gravity = np.array([
            np.sin(rpy[1]),  # pitch方向
            -np.sin(rpy[0]) * np.cos(rpy[1]),  # roll方向
            np.cos(rpy[0]) * np.cos(rpy[1])  # 垂直方向
        ])
        
        # 3. dof_pos (23维)
        dof_pos = joint_pos
        
        # 4. dof_vel (23维)
        dof_vel = joint_vel
        
        # 5. actions (23维) - 上一时刻的动作
        actions = self.last_actions
        
        # 6. gait_phase (2维) - sin和cos编码
        gait_phase = np.array([np.sin(self.gait_phase), np.cos(self.gait_phase)])
        
        # 更新历史
        self.obs_history['base_ang_vel'].append(base_ang_vel)
        self.obs_history['projected_gravity'].append(projected_gravity)
        self.obs_history['dof_pos'].append(dof_pos)
        self.obs_history['dof_vel'].append(dof_vel)
        self.obs_history['actions'].append(actions)
        self.obs_history['gait_phase'].append(gait_phase)
        
        # 保持历史长度
        for key in self.obs_history:
            if len(self.obs_history[key]) > self.history_length:
                self.obs_history[key].pop(0)
        
        # 构建观测字典（用于模型推理）
        obs_dict = {
            'base_ang_vel': torch.tensor(base_ang_vel, dtype=torch.float32).unsqueeze(0),
            'projected_gravity': torch.tensor(projected_gravity, dtype=torch.float32).unsqueeze(0),
            'dof_pos': torch.tensor(dof_pos, dtype=torch.float32).unsqueeze(0),
            'dof_vel': torch.tensor(dof_vel, dtype=torch.float32).unsqueeze(0),
            'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(0),
            'gait_phase': torch.tensor(gait_phase, dtype=torch.float32).unsqueeze(0),
        }
        
        # 添加历史观测
        if len(self.obs_history['base_ang_vel']) >= self.history_length:
            for key in ['base_ang_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'actions', 'gait_phase']:
                history = np.array(self.obs_history[key][-self.history_length:])
                obs_dict[f'short_history_{key}'] = torch.tensor(history.flatten(), dtype=torch.float32).unsqueeze(0)
        
        return obs_dict
    
    def compute_action(self, obs_dict):
        """
        使用策略模型计算动作
        
        Args:
            obs_dict: 观测字典
            
        Returns:
            actions: 23维动作向量
        """
        if self.actor_model is None:
            raise RuntimeError("模型未加载")
        
        with torch.no_grad():
            # 使用act_inference方法进行推理（返回均值，不采样）
            actions = self.actor_model.act_inference(obs_dict)
            actions = actions.cpu().numpy().flatten()
        
        return actions
    
    def send_control_command(self, actions):
        """
        将动作发送到机器人
        
        Args:
            actions: 23维动作向量（目标关节角度偏移）
        """
        # 将指定的自由度置0
        actions_zeroed = actions.copy()
        for idx in self.zero_action_indices:
            actions_zeroed[idx] = 0.0
        
        # 获取当前关节角度作为基准
        joint_pos = self.state_reader.get_joint_positions()
        
        # 计算目标关节角度 = 默认角度 + action_scale * actions
        # 这里假设actions是相对于默认位置的偏移
        default_joint_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # 左腿 (0-5)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # 右腿 (6-11)
            0.0, 0.0, 0.0,                   # 腰部 (12-14): waist_yaw, waist_roll(置0), waist_pitch(置0)
            0.0, 0.0, 0.0, 0.0,              # 左臂 (15-18): shoulder_pitch, roll(置0), yaw(置0), elbow
            0.0, 0.0, 0.0, 0.0               # 右臂 (19-22): shoulder_pitch, roll(置0), yaw(置0), elbow
        ])
        
        # 使用置0后的动作
        target_joint_pos = default_joint_pos + self.action_scale * actions_zeroed
        
        # 映射到29个电机索引
        # 注意：23维动作空间到29个电机的映射
        motor_indices = []
        action_indices = []
        
        # 腿部 (0-11) -> 电机 (0-11)
        motor_indices.extend(range(12))
        action_indices.extend(range(12))
        
        # 腰部 (12-14) -> 电机 (12-14)
        motor_indices.extend(range(12, 15))
        action_indices.extend(range(12, 15))
        
        # 左臂 (15-18) -> 电机 (15-18)
        motor_indices.extend(range(15, 19))
        action_indices.extend(range(15, 19))
        
        # 右臂 (19-22) -> 电机 (22-25)
        # 动作空间索引19-22对应电机索引22-25
        motor_indices.extend(range(22, 26))
        action_indices.extend(range(19, 23))
        
        # 设置控制命令
        for motor_idx, action_idx in zip(motor_indices, action_indices):
            self.low_cmd.motor_cmd[motor_idx].mode = 1  # 启用
            
            # 对于置0的自由度，保持默认位置
            if action_idx in self.zero_action_indices:
                # 保持默认位置，不跟随动作
                self.low_cmd.motor_cmd[motor_idx].q = default_joint_pos[action_idx]
            else:
                # 正常跟随动作
                self.low_cmd.motor_cmd[motor_idx].q = target_joint_pos[action_idx]
            
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            
            # PD参数：对于置0的自由度，使用较小的kp/kd以保持位置
            if action_idx in self.zero_action_indices:
                self.low_cmd.motor_cmd[motor_idx].kp = self.kp[action_idx] * 0.5  # 降低刚度
                self.low_cmd.motor_cmd[motor_idx].kd = self.kd[action_idx]
            else:
                self.low_cmd.motor_cmd[motor_idx].kp = self.kp[action_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kd[action_idx]
        
        # 更新步态相位
        self.gait_phase += self.phase_increment
        self.gait_phase = np.fmod(self.gait_phase, 2.0 * np.pi)
        
        # 计算CRC并发送
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
    
    def control_loop(self):
        """主控制循环"""
        print("\n🎮 开始控制循环...")
        print("按 Ctrl+C 停止\n")
        
        try:
            while True:
                start_time = time.time()
                
                # 构建观测
                obs_dict = self.build_observation()
                if obs_dict is None:
                    time.sleep(0.01)
                    continue
                
                # 计算动作
                actions = self.compute_action(obs_dict)
                self.last_actions = actions
                
                # 发送控制命令
                self.send_control_command(actions)
                
                # 控制频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.control_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"⚠️  控制循环超时: {elapsed:.4f}s > {self.control_dt:.4f}s")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 停止控制")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """运行部署"""
        print("="*80)
        print("G1机器人策略部署")
        print("="*80)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 初始化机器人连接
        self.init_robot_connection()
        
        # 3. 等待用户确认
        print("\n⚠️  警告: 机器人即将开始运动！")
        print("   请确保:")
        print("   1. 机器人周围没有障碍物")
        print("   2. 机器人处于安全位置")
        print("   3. 你已经准备好随时按 Ctrl+C 停止")
        input("\n按 Enter 继续，或 Ctrl+C 取消...")
        
        # 4. 开始控制循环
        self.control_loop()
        
        print("\n✅ 部署结束")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 deploy_policy.py <checkpoint_path> [network_interface]")
        print("\n示例:")
        print("  python3 deploy_policy.py ../../../../logs/BigStrideWalking/xxx/model_15000.pt wlp0s20f3")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    network_interface = sys.argv[2] if len(sys.argv) > 2 else None
    
    deployer = PolicyDeployer(checkpoint_path, network_interface)
    deployer.run()

