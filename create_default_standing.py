#!/usr/bin/env python3
"""
创建基于机器人默认关节角度的稳定站立姿态
这些角度是工程师设计的，确保机器人能够平衡站立
"""

import numpy as np
import joblib
from pathlib import Path
from omegaconf import OmegaConf

print("="*60)
print("创建机器人默认站立姿态")
print("="*60)

# 加载机器人配置
robot_config = OmegaConf.load("humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml")

# 获取默认关节角度（工程师设计的平衡姿态）
# default_joint_angles是字典，需要按照dof_names的顺序提取
dof_names = robot_config.robot.dof_names  # 这是关节名列表
default_joint_angles_dict = robot_config.robot.init_state.default_joint_angles

# 按顺序提取关节角度
default_dof_pos = []
for joint_name in dof_names:
    if joint_name in default_joint_angles_dict:
        default_dof_pos.append(default_joint_angles_dict[joint_name])
    else:
        print(f"  警告: {joint_name} 不在配置中，使用0")
        default_dof_pos.append(0.0)

default_dof_pos = np.array(default_dof_pos, dtype=np.float32)  # 使用float32而不是float64
num_dof = len(default_dof_pos)

print(f"\n📋 机器人配置:")
print(f"  DOF数量: {num_dof}")
print(f"  默认关节角度:")
print(f"    {default_dof_pos}")

# 设置合理的站立高度
target_height = 0.78  # 标准站立高度
root_pos = np.array([0.0, 0.0, target_height], dtype=np.float32)

# 完全直立的旋转（四元数 xyzw格式）
root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

# 所有速度为零（静止站立）
root_lin_vel = np.zeros(3, dtype=np.float32)
root_ang_vel = np.zeros(3, dtype=np.float32)
dof_vel = np.zeros(num_dof, dtype=np.float32)

# 为pose_aa创建简化的表示（全零）
# g1_29dof表示29个关节
num_joints_total = 29
pose_aa = np.zeros((num_joints_total, 3), dtype=np.float32)

# 生成多帧相同姿态（满足Motion Library的梯度计算要求）
num_frames = 10
fps = 30
dt = 1.0 / fps

print(f"\n🎯 生成站立姿态:")
print(f"  根位置: {root_pos}")
print(f"  根高度: {target_height}m")
print(f"  根旋转: {root_rot} (完全直立)")
print(f"  帧数: {num_frames}")
print(f"  帧率: {fps} fps")
print(f"  持续时间: {num_frames/fps:.2f}s")

# 构建motion数据（与Motion Library格式完全一致）
motion_data = {
    'motion0': {
        'root_trans_offset': np.tile(root_pos, (num_frames, 1)),
        'root_rot': np.tile(root_rot, (num_frames, 1)),
        'dof': np.tile(default_dof_pos, (num_frames, 1)),
        'pose_aa': np.tile(pose_aa, (num_frames, 1, 1)),
        'root_lin_vel': np.tile(root_lin_vel, (num_frames, 1)),
        'root_ang_vel': np.tile(root_ang_vel, (num_frames, 1)),
        'dof_vel': np.tile(dof_vel, (num_frames, 1)),
        'fps': fps,
        'motion_times': np.arange(num_frames) * dt,
    }
}

# 保存
output_file = "humanoidverse/data/motions/g1_29dof_anneal_23dof/default_standing.pkl"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(motion_data, output_file)

print(f"\n✅ 保存成功!")
print(f"   文件: {output_file}")

print(f"\n📊 数据验证:")
print(f"  root_trans_offset shape: {motion_data['motion0']['root_trans_offset'].shape}")
print(f"  root_rot shape: {motion_data['motion0']['root_rot'].shape}")
print(f"  dof shape: {motion_data['motion0']['dof'].shape}")
print(f"  pose_aa shape: {motion_data['motion0']['pose_aa'].shape}")

print(f"\n💡 使用方法:")
print(f"   修改训练脚本，将motion_file改为:")
print(f'   robot.motion.motion_file="{output_file}"')

print(f"\n🚀 训练命令示例:")
print(f"   ./train_standing_from_cr7.sh")
print(f"   (脚本会自动使用这个新文件)")

print("\n" + "="*60)
print("完成！这个姿态基于机器人设计，应该非常稳定！")
print("="*60)

