"""
大步行走环境 (Big Stride Walking Environment)
不使用参考轨迹，纯靠奖励函数训练机器人大步向前走
"""
import torch
import numpy as np
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import quat_apply


class BigStrideWalking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_stride_buffers()
        self.init_done = True
    
    def _init_buffers(self):
        super()._init_buffers()
        # 目标前进方向速度（固定向前）
        self.target_forward_vel = self.config.target_forward_velocity if hasattr(self.config, 'target_forward_velocity') else 1.0
        
    def _init_stride_buffers(self):
        """初始化大步行走相关的缓存"""
        # 记录上一步的脚部位置，用于计算步幅
        self.prev_left_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.prev_right_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        # 记录每只脚的抬起时刻和落地时刻
        self.left_foot_lift_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.right_foot_lift_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        # 步幅统计（用于奖励）
        self.left_stride_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.right_stride_buffer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 侧向偏移统计（用于惩罚"大风车"）
        self.left_lateral_deviation = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.right_lateral_deviation = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # 接触状态记录
        self.last_left_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_right_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 前进距离累计
        self.cumulative_forward_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        # 定义侧向方向（机器人坐标系的y轴）
        self.lateral_vec = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float, device=self.device)
        
        # ========== 侧向偏移追踪（用于终止条件）==========
        # 记录初始侧向位置（全局坐标系的y坐标）
        self.initial_lateral_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # ========== 相位控制系统 ==========
        # 步态相位 [0, 2π]，用于生成周期性的步态信号
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 目标步频 (Hz)，控制相位增长速度
        # 默认：0.667 Hz = 1.5秒/步（慢走）
        self.target_cadence = self.config.target_cadence if hasattr(self.config, 'target_cadence') else 0.667
        # 相位增量 = 2π * 频率 * dt
        self.phase_increment = 2.0 * np.pi * self.target_cadence * self.dt
        
    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        # 重置步幅相关的buffer
        self.prev_left_foot_pos[env_ids] = self.simulator._rigid_body_pos[env_ids, self.feet_indices[0], :]
        self.prev_right_foot_pos[env_ids] = self.simulator._rigid_body_pos[env_ids, self.feet_indices[1], :]
        self.left_stride_buffer[env_ids] = 0.0
        self.right_stride_buffer[env_ids] = 0.0
        self.left_lateral_deviation[env_ids] = 0.0
        self.right_lateral_deviation[env_ids] = 0.0
        self.last_left_contact[env_ids] = False
        self.last_right_contact[env_ids] = False
        self.cumulative_forward_dist[env_ids] = 0.0
        self.last_root_pos[env_ids] = self.simulator.robot_root_states[env_ids, :3]
        # 记录初始侧向位置（全局坐标系的y坐标）
        self.initial_lateral_pos[env_ids] = self.simulator.robot_root_states[env_ids, 1]
        # 重置相位（随机初始相位避免所有环境同步）
        self.gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * np.pi
        
    def _post_physics_step(self):
        super()._post_physics_step()
        self._update_gait_phase()
        self._update_stride_tracking()
    
    def _check_termination(self):
        """检查终止条件，添加朝向偏差和侧向偏移的终止"""
        super()._check_termination()
        
        # 添加朝向偏差终止条件
        if hasattr(self.config.termination, 'terminate_by_heading_deviation') and self.config.termination.terminate_by_heading_deviation:
            from isaac_utils.rotations import calc_heading_quat, quat_to_angle_axis
            
            # 计算当前朝向与初始朝向（前进方向）的偏差
            # robot_root_states 四元数格式是 xyzw，需要指定 w_last=True
            current_heading_quat = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
            # current_heading_quat 返回的也是 xyzw 格式
            # 计算角度偏差（yaw）- quat_to_angle_axis 期望 xyzw 格式，只接受一个参数
            angle, _ = quat_to_angle_axis(current_heading_quat)
            
            # 如果朝向偏离超过阈值，则终止
            max_heading_deviation = self.config.termination_scales.max_heading_deviation if hasattr(self.config.termination_scales, 'max_heading_deviation') else 1.57  # 默认90度
            self.reset_buf |= torch.abs(angle) > max_heading_deviation
        
        # 添加侧向偏移终止条件（防止横向"蛇形"走路）
        if hasattr(self.config.termination, 'terminate_by_lateral_deviation') and self.config.termination.terminate_by_lateral_deviation:
            # 计算当前侧向位置与初始位置的偏差（全局坐标系的y轴）
            current_lateral_pos = self.simulator.robot_root_states[:, 1]
            lateral_deviation = torch.abs(current_lateral_pos - self.initial_lateral_pos)
            
            # 如果侧向偏移超过阈值，则终止
            max_lateral_deviation = self.config.termination_scales.max_lateral_deviation if hasattr(self.config.termination_scales, 'max_lateral_deviation') else 1.0  # 默认1.0米
            self.reset_buf |= lateral_deviation > max_lateral_deviation
    
    def _update_gait_phase(self):
        """更新步态相位"""
        self.gait_phase += self.phase_increment
        # 保持相位在 [0, 2π] 范围内
        self.gait_phase = torch.fmod(self.gait_phase, 2.0 * np.pi)
    
    def _get_obs_gait_phase(self):
        """返回步态相位的 sin 和 cos 编码（周期性特征）"""
        # 使用 sin/cos 编码避免相位在 0/2π 处的不连续
        phase_sin = torch.sin(self.gait_phase).unsqueeze(1)  # [num_envs, 1]
        phase_cos = torch.cos(self.gait_phase).unsqueeze(1)  # [num_envs, 1]
        return torch.cat([phase_sin, phase_cos], dim=1)  # [num_envs, 2]
        
    def _update_stride_tracking(self):
        """更新步幅跟踪信息"""
        # 获取脚部接触状态
        left_contact = self.simulator.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.simulator.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        # 获取当前脚部位置
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0], :]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1], :]
        
        # 检测左脚从空中到接触地面（落地时刻）
        left_landing = left_contact & (~self.last_left_contact)
        # 检测右脚从空中到接触地面（落地时刻）
        right_landing = right_contact & (~self.last_right_contact)
        
        # 检测左脚从接触到离地（抬起时刻）
        left_liftoff = (~left_contact) & self.last_left_contact
        # 检测右脚从接触到离地（抬起时刻）
        right_liftoff = (~right_contact) & self.last_right_contact
        
        # 记录抬起时刻的位置
        self.left_foot_lift_pos[left_liftoff] = left_foot_pos[left_liftoff]
        self.right_foot_lift_pos[right_liftoff] = right_foot_pos[right_liftoff]
        
        # 计算步幅（抬起位置到落地位置的前向距离）和侧向偏移
        if left_landing.any():
            num_left = left_landing.sum().item()
            # forward_vec 可能是1D或2D
            if self.forward_vec.dim() == 1:
                forward_vec_left = self.forward_vec.unsqueeze(0).expand(num_left, -1)
            else:
                forward_vec_left = self.forward_vec[left_landing] if self.forward_vec.shape[0] == self.num_envs else self.forward_vec[0:1].expand(num_left, -1)
            
            # lateral_vec 始终是1D [3]
            lateral_vec_left = self.lateral_vec.unsqueeze(0).expand(num_left, -1)
            
            # 转换到世界坐标系
            forward_vec = quat_apply(self.base_quat[left_landing], forward_vec_left, w_last=True)
            lateral_vec = quat_apply(self.base_quat[left_landing], lateral_vec_left, w_last=True)
            
            stride_vec = left_foot_pos[left_landing] - self.left_foot_lift_pos[left_landing]
            # 投影到前进方向
            left_forward_stride = (stride_vec * forward_vec).sum(dim=-1)
            self.left_stride_buffer[left_landing] = left_forward_stride
            # 投影到侧向（惩罚"大风车"）
            left_lateral = torch.abs((stride_vec * lateral_vec).sum(dim=-1))
            self.left_lateral_deviation[left_landing] = left_lateral
            
        if right_landing.any():
            num_right = right_landing.sum().item()
            # forward_vec 可能是1D或2D
            if self.forward_vec.dim() == 1:
                forward_vec_right = self.forward_vec.unsqueeze(0).expand(num_right, -1)
            else:
                forward_vec_right = self.forward_vec[right_landing] if self.forward_vec.shape[0] == self.num_envs else self.forward_vec[0:1].expand(num_right, -1)
            
            # lateral_vec 始终是1D [3]
            lateral_vec_right = self.lateral_vec.unsqueeze(0).expand(num_right, -1)
            
            # 转换到世界坐标系
            forward_vec = quat_apply(self.base_quat[right_landing], forward_vec_right, w_last=True)
            lateral_vec = quat_apply(self.base_quat[right_landing], lateral_vec_right, w_last=True)
            
            stride_vec = right_foot_pos[right_landing] - self.right_foot_lift_pos[right_landing]
            right_forward_stride = (stride_vec * forward_vec).sum(dim=-1)
            self.right_stride_buffer[right_landing] = right_forward_stride
            # 投影到侧向（惩罚"大风车"）
            right_lateral = torch.abs((stride_vec * lateral_vec).sum(dim=-1))
            self.right_lateral_deviation[right_landing] = right_lateral
        
        # 更新累计前进距离
        root_displacement = self.simulator.robot_root_states[:, :3] - self.last_root_pos
        # forward_vec is [num_envs, 3] or needs to be broadcasted
        if self.forward_vec.dim() == 1:
            forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)
        else:
            forward_vec_expanded = self.forward_vec
        forward_vec = quat_apply(self.base_quat, forward_vec_expanded, w_last=True)
        forward_dist = (root_displacement * forward_vec).sum(dim=-1)
        self.cumulative_forward_dist += forward_dist
        self.last_root_pos[:] = self.simulator.robot_root_states[:, :3]
        
        # 更新接触状态
        self.last_left_contact[:] = left_contact
        self.last_right_contact[:] = right_contact
    
    ################ 奖励函数 ################
    
    def _reward_forward_velocity(self):
        """奖励向前的速度（极慢走：0.3-0.5 m/s）"""
        # 获取机器人当前朝向
        if self.forward_vec.dim() == 1:
            forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)
        else:
            forward_vec_expanded = self.forward_vec
        forward = quat_apply(self.base_quat, forward_vec_expanded, w_last=True)
        # 计算沿朝向的速度
        forward_vel = (self.base_lin_vel * forward).sum(dim=-1)
        # 目标速度（匹配1.5秒/步的慢节奏）
        target_vel = self.target_forward_vel  # 使用配置中的0.4 m/s
        # 速度误差
        vel_error = torch.abs(forward_vel - target_vel)
        # 指数奖励
        sigma = self.config.rewards.reward_tracking_sigma.get('forward_velocity', 0.5)
        return torch.exp(-vel_error / sigma)
    
    def _reward_big_stride(self):
        """奖励大步幅（慢走需要更大的步幅）"""
        # 取最近一次的步幅（左右脚的平均）
        avg_stride = (self.left_stride_buffer + self.right_stride_buffer) / 2.0
        # 鼓励步幅大于0.5m（慢走特征）
        target_stride = 0.5  # 目标步幅50cm
        stride_diff = torch.abs(avg_stride - target_stride)
        sigma = self.config.rewards.reward_tracking_sigma.get('big_stride', 0.15)
        return torch.exp(-stride_diff / sigma)
    
    def _reward_stride_symmetry(self):
        """奖励左右脚步幅对称"""
        stride_diff = torch.abs(self.left_stride_buffer - self.right_stride_buffer)
        sigma = self.config.rewards.reward_tracking_sigma.get('stride_symmetry', 0.15)
        return torch.exp(-stride_diff / sigma)
    
    def _reward_slow_cadence(self):
        """奖励低步频（慢走特征）- 基于相位控制"""
        # 相位已经在 _update_gait_phase 中按目标步频更新
        # 这里奖励机器人的实际步频与相位步频的匹配度
        
        # 估算实际步频：基于速度和步幅
        if self.forward_vec.dim() == 1:
            forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)
        else:
            forward_vec_expanded = self.forward_vec
        forward = quat_apply(self.base_quat, forward_vec_expanded, w_last=True)
        forward_vel = torch.abs((self.base_lin_vel * forward).sum(dim=-1))
        
        avg_stride = (self.left_stride_buffer + self.right_stride_buffer) / 2.0
        # 避免除零，初始步幅为0时使用较大值避免cadence过大
        avg_stride_safe = torch.clamp(avg_stride, min=0.2)
        
        # 估算实际步频（步/秒），限制在合理范围
        actual_cadence = torch.clamp(forward_vel / avg_stride_safe, max=10.0)
        
        # 奖励与目标步频匹配
        cadence_error = torch.abs(actual_cadence - self.target_cadence)
        # 限制误差避免exp参数过大
        cadence_error_clamped = torch.clamp(cadence_error, max=5.0)
        
        sigma = self.config.rewards.reward_tracking_sigma.get('slow_cadence', 0.5)
        return torch.exp(-cadence_error_clamped / sigma)
    
    def _reward_phase_foot_contact(self):
        """奖励脚部接触状态与相位信号同步"""
        # 左脚相位: [0, π] 抬起, [π, 2π] 着地
        # 右脚相位: [0, π] 着地, [π, 2π] 抬起（与左脚反相）
        
        # 获取接触状态
        left_contact = self.simulator.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.simulator.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        # 左脚期望接触（相位在 [π, 2π]）
        left_should_contact = self.gait_phase > np.pi
        # 右脚期望接触（相位在 [0, π]）
        right_should_contact = self.gait_phase <= np.pi
        
        # 计算匹配度（True=1.0, False=0.0）
        left_match = (left_contact == left_should_contact).float()
        right_match = (right_contact == right_should_contact).float()
        
        # 平均匹配度
        return (left_match + right_match) / 2.0
    
    def _reward_penalty_feet_lateral_motion(self):
        """惩罚脚部侧向运动（"大风车"式摆腿）"""
        # 获取脚部线速度（前3维）
        left_foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices[0], :3]   # [num_envs, 3]
        right_foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices[1], :3]  # [num_envs, 3]
        
        # 转换侧向向量到世界坐标系
        lateral_vec_expanded = self.lateral_vec.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        lateral_world = quat_apply(self.base_quat, lateral_vec_expanded, w_last=True)
        
        # 计算侧向速度分量
        left_lateral_vel = torch.abs((left_foot_vel * lateral_world).sum(dim=-1))
        right_lateral_vel = torch.abs((right_foot_vel * lateral_world).sum(dim=-1))
        
        # 平均侧向速度（越大惩罚越重）
        avg_lateral_vel = (left_lateral_vel + right_lateral_vel) / 2.0
        
        # 返回负值作为惩罚
        return -avg_lateral_vel
    
    def _reward_penalty_feet_lateral_deviation(self):
        """惩罚步幅的侧向偏移（落地时计算）"""
        # 使用已计算的侧向偏移
        avg_lateral_deviation = (self.left_lateral_deviation + self.right_lateral_deviation) / 2.0
        # 限制偏移值避免exp溢出，最大惩罚为 -exp(5) ≈ -148
        avg_lateral_deviation_clamped = torch.clamp(avg_lateral_deviation, max=0.5)  # 最大50cm
        # 返回负的指数惩罚
        return -torch.exp(avg_lateral_deviation_clamped / 0.1)
    
    def _reward_feet_distance(self):
        """奖励保持合理的两脚间距（不能太宽或太窄）"""
        # 获取两脚位置
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0], :]   # [num_envs, 3]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1], :]  # [num_envs, 3]
        
        # 计算两脚间的侧向距离（Y轴）
        lateral_vec_expanded = self.lateral_vec.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        lateral_world = quat_apply(self.base_quat, lateral_vec_expanded, w_last=True)
        
        # 计算两脚相对位置在侧向的投影
        feet_diff = right_foot_pos - left_foot_pos
        lateral_distance = torch.abs((feet_diff * lateral_world).sum(dim=-1))
        
        # 理想侧向距离：0.15-0.25m（髋宽左右）
        # 太窄(<0.1m)：不稳定
        # 太宽(>0.35m)：不自然、可能是"大风车"
        target_distance = 0.20  # 目标20cm
        distance_error = torch.abs(lateral_distance - target_distance)
        
        # 指数奖励
        return torch.exp(-distance_error / 0.1)
    
    def _reward_penalty_feet_distance_violation(self):
        """惩罚两脚距离超出安全范围"""
        # 获取两脚位置
        left_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[0], :]
        right_foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices[1], :]
        
        # 计算两脚间的侧向距离
        lateral_vec_expanded = self.lateral_vec.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        lateral_world = quat_apply(self.base_quat, lateral_vec_expanded, w_last=True)
        
        feet_diff = right_foot_pos - left_foot_pos
        lateral_distance = torch.abs((feet_diff * lateral_world).sum(dim=-1))
        
        # 惩罚过窄（<0.08m）或过宽（>0.40m）
        too_narrow = torch.clamp(0.08 - lateral_distance, min=0.0)  # 窄于8cm
        too_wide = torch.clamp(lateral_distance - 0.40, min=0.0)    # 宽于40cm
        
        # 返回惩罚（越违规惩罚越大）
        return -(too_narrow + too_wide) * 10.0
    
    def _reward_upright_posture(self):
        """奖励保持直立"""
        # 使用投影重力判断倾斜
        return torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1) / 0.15)
    
    def _reward_feet_clearance(self):
        """奖励抬脚高度（摆动腿离地）"""
        left_contact = self.simulator.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.simulator.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        left_foot_height = self.simulator._rigid_body_pos[:, self.feet_indices[0], 2]
        right_foot_height = self.simulator._rigid_body_pos[:, self.feet_indices[1], 2]
        
        # 只奖励悬空的脚（且高度在合理范围内）
        left_clearance = torch.clamp(left_foot_height - 0.05, 0.0, 0.20) * (~left_contact).float()
        right_clearance = torch.clamp(right_foot_height - 0.05, 0.0, 0.20) * (~right_contact).float()
        
        return (left_clearance + right_clearance) * 2.0
    
    def _reward_feet_contact_alternation(self):
        """奖励双脚交替接触（避免双脚同时离地或同时着地）"""
        left_contact = self.simulator.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.simulator.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        # 理想：左右脚一个接触、一个悬空
        # 惩罚：两个都接触 或 两个都悬空
        both_contact = (left_contact & right_contact).float()
        both_air = (~left_contact & ~right_contact).float()
        alternation = 1.0 - both_contact - both_air
        return alternation
    
    def _reward_heading_stability(self):
        """奖励朝向稳定（保持向前，不左右摇摆）"""
        # 身体朝向应该与前进方向一致
        if self.forward_vec.dim() == 1:
            forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)
        else:
            forward_vec_expanded = self.forward_vec
        forward = quat_apply(self.base_quat, forward_vec_expanded, w_last=True)
        # 计算与x轴的夹角（假设初始朝向为x正方向）
        heading_error = torch.abs(torch.atan2(forward[:, 1], forward[:, 0]))
        return torch.exp(-heading_error / 0.3)
    
    def _reward_low_torque(self):
        """轻微惩罚过大的扭矩（鼓励高效运动）"""
        return torch.sum(torch.square(self.torques), dim=-1)
    
    def _reward_smooth_motion(self):
        """奖励平滑运动（惩罚动作突变）"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
    
    def _reward_base_height(self):
        """奖励保持合理的身体高度"""
        target_height = 0.70  # 根据G1机器人调整
        height_error = torch.abs(self.simulator.robot_root_states[:, 2] - target_height)
        return torch.exp(-height_error / 0.15)
    
    def _reward_low_base_motion(self):
        """惩罚身体在非前进方向的运动（侧向、上下晃动）"""
        # 侧向速度
        lateral_vel = torch.abs(self.base_lin_vel[:, 1])
        # 垂直速度
        vertical_vel = torch.abs(self.base_lin_vel[:, 2])
        return lateral_vel + vertical_vel * 2.0
    
    def _reward_forward_progress(self):
        """奖励整体前进距离"""
        # 每个时间步的前进距离
        root_displacement = self.simulator.robot_root_states[:, :3] - self.last_root_pos
        if self.forward_vec.dim() == 1:
            forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)
        else:
            forward_vec_expanded = self.forward_vec
        forward_vec = quat_apply(self.base_quat, forward_vec_expanded, w_last=True)
        forward_dist = (root_displacement * forward_vec).sum(dim=-1)
        # 只奖励正向前进
        return torch.clamp(forward_dist, min=0.0) * 50.0
    
    def _reward_penalty_lin_vel_z(self):
        """惩罚垂直方向的速度（应该只水平移动）"""
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        """惩罚pitch和roll方向的角速度（保持直立）"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)
    
    def _reward_penalty_ang_vel_z(self):
        """惩罚yaw方向的角速度（保持直线前进）"""
        # base_ang_vel[:, 2] 是绕Z轴的角速度（yaw）
        return torch.square(self.base_ang_vel[:, 2])
    
    def _reward_penalty_orientation(self):
        """惩罚身体倾斜（使用投影重力）"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
    
    def _reward_penalty_dof_acc(self):
        """惩罚关节加速度"""
        return torch.sum(torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=-1)
    
    def _reward_penalty_action_rate(self):
        """惩罚动作变化率（与smooth_motion相同）"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
    
    def _reward_limits_dof_pos(self):
        """惩罚关节位置接近极限"""
        # 软限制
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=-1)
    
    def _reward_limits_dof_vel(self):
        """惩罚关节速度接近极限"""
        return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * 0.95).clip(min=0., max=1.), dim=-1)
    
    def _reward_limits_torque(self):
        """惩罚扭矩接近极限"""
        return torch.sum((torch.abs(self.torques) - self.torque_limits * 0.90).clip(min=0.), dim=-1)
    
    def _reward_penalty_arm_motion(self):
        """惩罚手臂过度运动（保持手臂相对静止）"""
        # 获取手臂关节的索引（从15开始是上肢关节：shoulder*3 + elbow，共8个）
        # dof_names: [..., 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #              'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        arm_start_idx = 15  # 上肢关节从索引15开始
        arm_end_idx = 23    # 到索引22（共8个上肢关节）
        
        # 惩罚手臂关节的速度（鼓励保持静止）
        arm_dof_vel = self.simulator.dof_vel[:, arm_start_idx:arm_end_idx]
        arm_vel_penalty = torch.sum(torch.square(arm_dof_vel), dim=-1)
        
        # 惩罚手臂关节偏离默认位置（默认位置为0）
        arm_dof_pos = self.simulator.dof_pos[:, arm_start_idx:arm_end_idx]
        arm_pos_penalty = torch.sum(torch.square(arm_dof_pos - self.default_dof_pos[:, arm_start_idx:arm_end_idx]), dim=-1)
        
        # 组合两种惩罚
        return arm_vel_penalty + 0.5 * arm_pos_penalty

