"""
Standing Task for Humanoid Robot
训练机器人保持稳定站立的任务
"""

import torch
import numpy as np
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import quat_rotate_inverse
from humanoidverse.utils.torch_utils import torch_rand_float
from loguru import logger


class LeggedRobotStanding(LeggedRobotBase):
    """
    Standing task: 训练人形机器人保持稳定站立
    
    核心目标:
    1. 保持直立姿态
    2. 保持目标高度
    3. 双脚保持合理间距
    4. 全身保持静止（速度接近零）
    5. 重心稳定
    """
    
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        logger.info("Standing Task Initialized")
        logger.info(f"Target base height: {self.config.target_base_height:.3f}m")
        logger.info(f"Target feet distance: {self.config.target_feet_distance:.3f}m")
        self.init_done = True
    
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        # Standing任务不需要复杂的命令，但保留接口兼容性
        self.standing_target_height = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        ) * self.config.target_base_height
        
        # 记录初始状态用于奖励计算
        self.initial_base_height = torch.zeros(self.num_envs, device=self.device)
        self.feet_contact_history = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )

    def _check_termination(self):
        """检查终止条件"""
        super()._check_termination()
        
        # 额外的站立任务终止条件
        if self.config.termination.terminate_by_low_height:
            base_height = self.simulator.robot_root_states[:, 2]
            self.reset_buf |= (base_height < self.config.termination_scales.termination_min_base_height)
            self.reset_buf |= (base_height > self.config.termination_scales.termination_max_base_height)
        
        if self.config.termination.terminate_by_high_tilt:
            # 检查倾斜角度是否过大
            base_tilt = torch.abs(self.projected_gravity[:, :2])
            self.reset_buf |= torch.any(
                base_tilt > self.config.termination_scales.termination_max_base_tilt, 
                dim=1
            )

    def _reset_tasks_callback(self, env_ids):
        """重置任务回调"""
        super()._reset_tasks_callback(env_ids)
        # 记录初始高度
        self.initial_base_height[env_ids] = self.simulator.robot_root_states[env_ids, 2]

    def set_is_evaluating(self):
        """设置为评估模式"""
        super().set_is_evaluating()
        logger.info("Standing task set to evaluation mode")

    ########################### 核心站立奖励 ###########################
    
    def _reward_standing_upright(self):
        """奖励保持直立姿态 - 最重要的奖励"""
        # 计算身体与垂直方向的偏离（使用投影重力）
        # projected_gravity在完全直立时应该是[0, 0, -1]
        upright_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # 使用指数奖励，越接近垂直奖励越高
        return torch.exp(-upright_error / self.config.rewards.reward_tracking_sigma.orientation)
    
    def _reward_standing_base_height(self):
        """奖励保持目标高度"""
        base_height = self.simulator.robot_root_states[:, 2]
        height_error = torch.square(base_height - self.config.rewards.desired_base_height)
        return torch.exp(-height_error / self.config.rewards.reward_tracking_sigma.base_height)
    
    def _reward_standing_feet_placement(self):
        """奖励保持合理的双脚间距"""
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        
        # 目标是保持desired_feet_distance的距离
        distance_error = torch.square(
            feet_distance - self.config.rewards.desired_feet_distance
        )
        return torch.exp(-distance_error / self.config.rewards.reward_tracking_sigma.feet_distance)
    
    def _reward_standing_base_stability(self):
        """奖励质心稳定（xy平面上的位移最小）"""
        # 惩罚质心在xy平面的移动
        base_pos_xy = self.simulator.robot_root_states[:, :2]
        # 计算相对于环境原点的位移
        displacement = torch.sum(torch.square(base_pos_xy), dim=1)
        # 希望质心尽量保持在原地
        return torch.exp(-displacement / 0.01)  # sigma=0.01m
    
    def _reward_standing_zero_velocity(self):
        """奖励保持静止（线速度和角速度都接近零）"""
        # 线速度奖励
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / self.config.rewards.reward_tracking_sigma.velocity)
        
        # 角速度奖励
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel), dim=1)
        ang_vel_reward = torch.exp(-ang_vel_error / self.config.rewards.reward_tracking_sigma.velocity)
        
        # 综合奖励
        return (lin_vel_reward + ang_vel_reward) / 2.0
    
    ########################### 姿态优化奖励 ###########################
    
    def _reward_standing_symmetric_stance(self):
        """奖励对称站姿（左右脚高度和位置对称）"""
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0]]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1]]
        
        # 高度应该相同
        height_diff = torch.abs(left_foot_pos[:, 2] - right_foot_pos[:, 2])
        
        # y方向（左右）应该对称
        y_diff = torch.abs(left_foot_pos[:, 1] + right_foot_pos[:, 1])  # 理想情况下和为0
        
        symmetry_error = height_diff + y_diff
        return torch.exp(-symmetry_error / 0.05)  # sigma=0.05m
    
    def _reward_standing_joint_default(self):
        """奖励关节角度接近默认值（站立姿态）"""
        joint_deviation = torch.sum(
            torch.square(self.simulator.dof_pos - self.default_dof_pos),
            dim=1
        )
        return torch.exp(-joint_deviation / self.config.rewards.reward_tracking_sigma.joint_pos)
    
    ########################### 物理约束惩罚 ###########################
    
    def _reward_penalty_orientation(self):
        """惩罚身体倾斜"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_penalty_lin_vel(self):
        """惩罚线速度（站立时应该静止）"""
        return torch.sum(torch.square(self.base_lin_vel), dim=1)
    
    def _reward_penalty_ang_vel(self):
        """惩罚角速度（站立时应该静止）"""
        return torch.sum(torch.square(self.base_ang_vel), dim=1)
    
    def _reward_penalty_base_height_deviation(self):
        """惩罚高度偏离目标"""
        base_height = self.simulator.robot_root_states[:, 2]
        deviation = torch.abs(base_height - self.config.rewards.desired_base_height)
        # 在容忍度内不惩罚
        return torch.clamp(
            deviation - self.config.rewards.base_height_tolerance, 
            min=0.0
        )
    
    ########################### 脚部接触惩罚 ###########################
    
    def _reward_penalty_feet_contact_forces(self):
        """惩罚过大的脚部接触力"""
        # 站立时双脚应该平稳接触地面，接触力不应过大
        max_force = 300.0  # N
        contact_forces = torch.norm(
            self.simulator.contact_forces[:, self.feet_indices, :], 
            dim=-1
        )
        return torch.sum(
            torch.clamp(contact_forces - max_force, min=0.0), 
            dim=1
        )
    
    def _reward_penalty_uneven_feet_contact(self):
        """惩罚双脚接触力不均（站立时应该均匀分布）"""
        contact_forces = torch.norm(
            self.simulator.contact_forces[:, self.feet_indices, :], 
            dim=-1
        )
        left_force = contact_forces[:, 0]
        right_force = contact_forces[:, 1]
        
        # 计算接触力差异（理想情况下应该相等）
        force_diff = torch.abs(left_force - right_force)
        # 归一化到总力
        total_force = left_force + right_force + 1e-6  # 避免除零
        normalized_diff = force_diff / total_force
        
        return normalized_diff

    ########################### 观测函数 ###########################
    
    def _get_obs_standing_target_height(self):
        """返回目标高度观测"""
        return self.standing_target_height.unsqueeze(1)
    
    def _get_obs_feet_distance(self):
        """返回当前双脚距离"""
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)
        return feet_distance.unsqueeze(1)

