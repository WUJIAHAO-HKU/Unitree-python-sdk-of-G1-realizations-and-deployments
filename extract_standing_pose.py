"""
从Motion Tracking的参考动作文件中提取特定时刻的站立姿态
用法：python extract_standing_pose.py --motion_file <path> --time 2.8 --output standing_pose.pkl
"""

import torch
import pickle
import joblib
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as sRot

def extract_standing_pose_from_motion(motion_file, target_time=3.0, output_file="standing_pose.pkl"):
    """
    从motion文件中提取特定时刻的姿态数据
    
    Args:
        motion_file: 参考动作文件路径（.pkl）
        target_time: 目标时刻（秒），例如2.8秒时CR7处于站立姿态
        output_file: 输出文件路径
    """
    
    print(f"[1/4] 加载动作文件: {motion_file}")
    # Motion Library使用joblib而不是pickle
    motion_data = joblib.load(motion_file)
    
    # 动作文件的结构可能是：
    # 1. {'motion0': {...}, 'motion1': {...}, ...}
    # 2. {'long_motion_name': {...}}
    # 3. 直接是motion数据（较少见）
    
    # 获取第一个motion
    if isinstance(motion_data, dict):
        motion_keys = list(motion_data.keys())
        if len(motion_keys) > 0:
            first_key = motion_keys[0]
            motion = motion_data[first_key]
            print(f"  检测到motion键: '{first_key}'")
        else:
            motion = motion_data
            print(f"  使用整个数据作为motion")
    else:
        motion = motion_data
        print(f"  使用整个数据作为motion")
    
    print(f"\n[2/4] 分析动作数据结构:")
    print(f"  可用的键: {motion.keys()}")
    
    # 显示每个键的数据类型和形状
    for key in motion.keys():
        value = motion[key]
        if isinstance(value, np.ndarray):
            print(f"    {key}: numpy array, shape={value.shape}")
        else:
            print(f"    {key}: {type(value).__name__}")
    
    # 获取帧率和总时长
    if 'fps' in motion:
        fps = motion['fps']
    else:
        fps = 50  # 默认50Hz
    print(f"  帧率 (fps): {fps}")
    
    # 计算目标帧
    target_frame = int(target_time * fps)
    print(f"  目标时刻: {target_time}秒 → 第 {target_frame} 帧")
    
    # 提取该帧的数据
    print(f"\n[3/4] 提取第 {target_frame} 帧的姿态数据:")
    standing_pose = {}
    
    # 提取关键数据
    data_keys = ['root_trans_offset', 'root_rot', 'dof', 'pose_aa', 
                 'root_lin_vel', 'root_ang_vel', 'dof_vel']
    
    for key in data_keys:
        if key in motion:
            data = motion[key]
            if isinstance(data, np.ndarray):
                # 数据形状通常是 (T, ...) 或 (T, num_features)
                if target_frame < data.shape[0]:
                    standing_pose[key] = data[target_frame]
                    print(f"  ✓ {key}: shape {data[target_frame].shape}")
                else:
                    print(f"  ✗ {key}: 目标帧超出范围 (max={data.shape[0]-1})")
            else:
                print(f"  ⚠ {key}: 非numpy数组，跳过")
        else:
            print(f"  - {key}: 不存在")
    
    # 为站立姿态补充缺失的速度数据（全为0）
    if 'root_lin_vel' not in standing_pose:
        standing_pose['root_lin_vel'] = np.zeros(3)
        print(f"  ℹ 添加零速度: root_lin_vel = zeros(3)")
    
    if 'root_ang_vel' not in standing_pose:
        standing_pose['root_ang_vel'] = np.zeros(3)
        print(f"  ℹ 添加零速度: root_ang_vel = zeros(3)")
    
    if 'dof_vel' not in standing_pose and 'dof' in standing_pose:
        num_dof = standing_pose['dof'].shape[0]
        standing_pose['dof_vel'] = np.zeros(num_dof)
        print(f"  ℹ 添加零速度: dof_vel = zeros({num_dof})")
    
    # 保存为新的motion文件格式
    # 注意：为了让Motion Library能够计算梯度，我们需要至少2帧
    # 这里复制站立姿态为连续的多帧（例如10帧，持续0.3秒）
    num_frames = 300
    dt = 1.0 / fps
    
    print(f"\n[4/4] 保存站立姿态到: {output_file}")
    print(f"  生成 {num_frames} 帧相同姿态（持续 {num_frames * dt:.2f}秒）以满足Motion Library要求")
    
    output_data = {
        'motion0': {
            'root_trans_offset': np.tile(standing_pose['root_trans_offset'], (num_frames, 1)) if 'root_trans_offset' in standing_pose else None,
            'root_rot': np.tile(standing_pose['root_rot'], (num_frames, 1)) if 'root_rot' in standing_pose else None,
            'dof': np.tile(standing_pose['dof'], (num_frames, 1)) if 'dof' in standing_pose else None,
            'pose_aa': np.tile(standing_pose['pose_aa'], (num_frames, 1, 1)) if 'pose_aa' in standing_pose else None,
            'root_lin_vel': np.tile(standing_pose['root_lin_vel'], (num_frames, 1)) if 'root_lin_vel' in standing_pose else None,
            'root_ang_vel': np.tile(standing_pose['root_ang_vel'], (num_frames, 1)) if 'root_ang_vel' in standing_pose else None,
            'dof_vel': np.tile(standing_pose['dof_vel'], (num_frames, 1)) if 'dof_vel' in standing_pose else None,
            'fps': fps,
            'motion_times': np.arange(num_frames) * dt,  # 时间序列: 0, dt, 2*dt, ...
        }
    }
    
    # 使用joblib保存，与Motion Library一致
    joblib.dump(output_data, output_file)
    
    print(f"  ✓ 保存成功!")
    print(f"\n" + "="*60)
    print(f"站立姿态提取完成！")
    print(f"输出文件: {output_file}")
    print(f"来源: {motion_file} @ {target_time}秒")
    print(f"="*60)
    
    return output_data


def visualize_standing_pose(standing_pose_file):
    """可选：可视化站立姿态数据"""
    print(f"\n[可视化] 检查站立姿态数据:")
    data = joblib.load(standing_pose_file)
    
    motion = data['motion0']
    print(f"  根位置: {motion['root_trans_offset']}")
    print(f"  根旋转: {motion['root_rot']}")
    print(f"  关节角度 (前5个): {motion['dof'][0, :5]}")
    print(f"  数据形状: dof={motion['dof'].shape}, pose_aa={motion['pose_aa'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Motion文件中提取站立姿态")
    parser.add_argument("--motion_file", type=str, required=True,
                        help="输入的motion文件路径 (.pkl)")
    parser.add_argument("--time", type=float, default=2.8,
                        help="提取姿态的时刻（秒），默认2.8秒")
    parser.add_argument("--output", type=str, default="standing_pose.pkl",
                        help="输出文件路径，默认 standing_pose.pkl")
    parser.add_argument("--visualize", action="store_true",
                        help="提取后可视化数据")
    
    args = parser.parse_args()
    
    # 提取站立姿态
    standing_pose = extract_standing_pose_from_motion(
        args.motion_file,
        args.time,
        args.output
    )
    
    # 可视化
    if args.visualize:
        visualize_standing_pose(args.output)
    
    print(f"\n💡 使用方法:")
    print(f"   在robot配置中设置: robot.motion.motion_file=\"{args.output}\"")
    print(f"   然后运行Motion Tracking任务，机器人会跟踪这个站立姿态")

