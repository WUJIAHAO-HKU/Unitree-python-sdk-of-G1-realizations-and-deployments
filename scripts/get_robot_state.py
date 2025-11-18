"""
获取G1机器人真实状态
用于查看机器人当前的关节角度、速度、IMU等传感器数据
"""
import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# G1机器人关节索引定义（23自由度版本）
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

# 关节名称（对应23自由度）
JOINT_NAMES = [
    # 左腿
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", 
    "left_ankle_pitch", "left_ankle_roll",
    # 右腿
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    # 腰部
    "waist_yaw", "waist_roll", "waist_pitch",
    # 左臂
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    # 右臂
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
]

class RobotStateReader:
    def __init__(self):
        self.low_state = None
        self.state_received = False
        self.counter = 0
        
    def LowStateHandler(self, msg: LowState_):
        """状态回调函数，每次收到新状态时自动调用"""
        self.low_state = msg
        self.state_received = True
        self.counter += 1
        
    def Init(self, network_interface=None):
        """初始化订阅者"""
        # 初始化通道工厂
        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)
        
        # 创建状态订阅者
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        print("✅ 已连接到机器人，等待状态数据...")
        
    def get_joint_positions(self):
        """获取所有关节角度（弧度）- 23自由度版本"""
        if self.low_state is None:
            return None
        
        # G1有29个电机，23自由度版本的有效索引映射：
        # 0-11: 左右腿 (12个)
        # 12-14: 腰部 (3个，但waist_roll和waist_pitch可能无效)
        # 15-18: 左臂 (4个)
        # 22-25: 右臂 (4个，跳过19-21手腕)
        joint_pos = []
        
        # 腿部 (0-11)
        for i in range(12):
            joint_pos.append(self.low_state.motor_state[i].q)
        
        # 腰部 (12-14)
        for i in range(12, 15):
            joint_pos.append(self.low_state.motor_state[i].q)
        
        # 左臂 (15-18)
        for i in range(15, 19):
            joint_pos.append(self.low_state.motor_state[i].q)
        
        # 右臂 (22-25，跳过19-21手腕)
        for i in range(22, 26):
            joint_pos.append(self.low_state.motor_state[i].q)
        
        return np.array(joint_pos)
    
    def get_joint_velocities(self):
        """获取所有关节速度（弧度/秒）- 23自由度版本"""
        if self.low_state is None:
            return None
        
        joint_vel = []
        
        # 腿部 (0-11)
        for i in range(12):
            joint_vel.append(self.low_state.motor_state[i].dq)
        
        # 腰部 (12-14)
        for i in range(12, 15):
            joint_vel.append(self.low_state.motor_state[i].dq)
        
        # 左臂 (15-18)
        for i in range(15, 19):
            joint_vel.append(self.low_state.motor_state[i].dq)
        
        # 右臂 (22-25)
        for i in range(22, 26):
            joint_vel.append(self.low_state.motor_state[i].dq)
        
        return np.array(joint_vel)
    
    def get_joint_torques(self):
        """获取所有关节力矩（Nm）- 23自由度版本"""
        if self.low_state is None:
            return None
        
        joint_tau = []
        
        # 腿部 (0-11)
        for i in range(12):
            joint_tau.append(self.low_state.motor_state[i].tau_est)
        
        # 腰部 (12-14)
        for i in range(12, 15):
            joint_tau.append(self.low_state.motor_state[i].tau_est)
        
        # 左臂 (15-18)
        for i in range(15, 19):
            joint_tau.append(self.low_state.motor_state[i].tau_est)
        
        # 右臂 (22-25)
        for i in range(22, 26):
            joint_tau.append(self.low_state.motor_state[i].tau_est)
        
        return np.array(joint_tau)
    
    def get_imu_data(self):
        """获取IMU数据"""
        if self.low_state is None:
            return None
        
        imu = self.low_state.imu_state
        return {
            'rpy': np.array([imu.rpy[0], imu.rpy[1], imu.rpy[2]]),  # Roll, Pitch, Yaw (弧度)
            'gyroscope': np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]]),  # 角速度 (rad/s)
            'accelerometer': np.array([imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2]]),  # 加速度 (m/s²)
            'quaternion': np.array([imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3]])  # 四元数
        }
    
    def get_battery_voltage(self):
        """获取电池电压"""
        if self.low_state is None:
            return None
        # 部分固件版本未暴露 bms_state，需容错
        if hasattr(self.low_state, "bms_state"):
            soc = getattr(self.low_state.bms_state, "SOC", None)
            return soc
        return None
    
    def print_state(self):
        """打印当前状态"""
        if not self.state_received:
            print("⏳ 等待机器人状态数据...")
            return
        
        print("\n" + "="*80)
        print(f"📊 机器人状态 (第 {self.counter} 次更新)")
        print("="*80)
        
        # IMU数据
        imu = self.get_imu_data()
        print("\n🧭 IMU数据:")
        print(f"  姿态角 (RPY): [{imu['rpy'][0]:.4f}, {imu['rpy'][1]:.4f}, {imu['rpy'][2]:.4f}] rad")
        print(f"  角速度: [{imu['gyroscope'][0]:.4f}, {imu['gyroscope'][1]:.4f}, {imu['gyroscope'][2]:.4f}] rad/s")
        print(f"  加速度: [{imu['accelerometer'][0]:.4f}, {imu['accelerometer'][1]:.4f}, {imu['accelerometer'][2]:.4f}] m/s²")
        
        # 关节角度
        joint_pos = self.get_joint_positions()
        print("\n🦵 关节角度 (rad):")
        print("  左腿: ", end="")
        for i in range(6):
            print(f"{JOINT_NAMES[i]}: {joint_pos[i]:.4f}  ", end="")
        print("\n  右腿: ", end="")
        for i in range(6, 12):
            print(f"{JOINT_NAMES[i]}: {joint_pos[i]:.4f}  ", end="")
        print("\n  腰部: ", end="")
        for i in range(12, 15):
            print(f"{JOINT_NAMES[i]}: {joint_pos[i]:.4f}  ", end="")
        print("\n  左臂: ", end="")
        for i in range(15, 19):
            print(f"{JOINT_NAMES[i]}: {joint_pos[i]:.4f}  ", end="")
        print("\n  右臂: ", end="")
        for i in range(19, 23):
            print(f"{JOINT_NAMES[i]}: {joint_pos[i]:.4f}  ", end="")
        
        # 关节速度
        joint_vel = self.get_joint_velocities()
        print("\n\n⚡ 关节速度 (rad/s):")
        print(f"  左腿: {joint_vel[0:6]}")
        print(f"  右腿: {joint_vel[6:12]}")
        print(f"  腰部: {joint_vel[12:15]}")
        print(f"  左臂: {joint_vel[15:19]}")
        print(f"  右臂: {joint_vel[19:23]}")
        
        # 关节力矩
        joint_tau = self.get_joint_torques()
        print("\n💪 关节力矩 (Nm):")
        print(f"  左腿: {joint_tau[0:6]}")
        print(f"  右腿: {joint_tau[6:12]}")
        print(f"  腰部: {joint_tau[12:15]}")
        print(f"  左臂: {joint_tau[15:19]}")
        print(f"  右臂: {joint_tau[19:23]}")
        
        # 电池
        battery = self.get_battery_voltage()
        if battery is not None:
            print(f"\n🔋 电池电量: {battery}%")
        else:
            print("\n🔋 电池电量: 当前固件未提供 bms_state")
        
        print("="*80 + "\n")
    
    def get_observation_vector(self):
        """
        获取用于策略推理的观测向量
        格式与仿真环境中的观测空间对应
        """
        if not self.state_received:
            return None
        
        imu = self.get_imu_data()
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        
        # 构建观测向量（根据你的观测空间配置调整）
        obs = []
        
        # 1. 投影重力 (3维)
        # 注意：这里需要根据实际IMU数据计算投影重力
        # 简化版本：使用RPY角度
        obs.extend(imu['rpy'].tolist())
        
        # 2. 关节角度 (23维)
        obs.extend(joint_pos.tolist())
        
        # 3. 关节速度 (23维)
        obs.extend(joint_vel.tolist())
        
        # 4. 角速度 (3维)
        obs.extend(imu['gyroscope'].tolist())
        
        return np.array(obs)

if __name__ == '__main__':
    print("="*80)
    print("G1机器人状态获取工具")
    print("="*80)
    print("\n使用方法:")
    print("  python3 get_robot_state.py [网络接口名称]")
    print("\n例如:")
    print("  python3 get_robot_state.py wlp0s20f3")
    print("\n提示: 使用 'ifconfig' 或 'ip addr' 查看网络接口名称")
    print("="*80 + "\n")
    
    # 获取网络接口名称
    if len(sys.argv) > 1:
        network_interface = sys.argv[1]
        print(f"📡 使用网络接口: {network_interface}")
    else:
        network_interface = None
        print("⚠️  未指定网络接口，使用默认设置")
    
    # 创建状态读取器
    reader = RobotStateReader()
    reader.Init(network_interface)
    
    # 主循环：每1秒打印一次状态
    try:
        print("\n按 Ctrl+C 退出\n")
        while True:
            time.sleep(1.0)
            reader.print_state()
    except KeyboardInterrupt:
        print("\n\n👋 已退出")
        sys.exit(0)

