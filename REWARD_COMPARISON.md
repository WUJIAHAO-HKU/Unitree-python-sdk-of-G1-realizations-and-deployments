# 站立 vs 走路奖励函数对比

## 📊 概述

本文档对比了专门为**站立姿态**和**走路动作**优化的两套奖励函数配置。

### 配置文件
- **站立**: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_standing.yaml`
- **走路**: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_walking.yaml`

---

## 🎯 核心设计理念

### 站立姿态 (Standing)
> **目标**: 保持静态稳定，精确匹配单帧姿态

**关键策略**:
- ❌ **严格惩罚**速度和动作变化
- ✅ **高度奖励**位置精确度
- ✅ **鼓励**静止不动
- ⚠️ **容忍**较大的姿态偏差（因为是静态）

### 走路动作 (Walking)
> **目标**: 动态跟踪运动轨迹，连续执行动作序列

**关键策略**:
- ✅ **高度奖励**速度匹配和关节运动
- ✅ **鼓励**连续流畅的动作
- ✅ **严格要求**轨迹跟踪精度
- ⚠️ **允许**必要的力量和扭矩

---

## 📈 关键参数对比

### 1. Tracking Sigma（跟踪容差）

**Sigma越小 = 要求越精确；Sigma越大 = 容忍偏差越大**

| 参数 | 站立 (Standing) | 走路 (Walking) | 说明 |
|------|----------------|---------------|------|
| `teleop_upper_body_pos` | 0.2 | 0.08 | 走路要求上身位置更精确 |
| `teleop_lower_body_pos` | 0.5 | 0.12 | 走路要求下身位置更精确 |
| `teleop_feet_pos` | 0.15 | 0.08 | 走路要求脚部位置更精确 |
| `teleop_body_vel` | **0.2** | **1.5** | ⚡ 关键差异！站立要求速度≈0，走路鼓励速度匹配 |
| `teleop_body_ang_vel` | **0.2** | **1.5** | ⚡ 关键差异！站立要求角速度≈0，走路允许转动 |
| `teleop_joint_pos` | 3.0 | 1.5 | 走路要求关节位置更精确 |
| `teleop_joint_vel` | **0.5** | **1.5** | ⚡ 关键差异！站立要求关节静止，走路鼓励关节运动 |

### 2. 奖励权重对比

| 奖励项 | 站立 | 走路 | 说明 |
|--------|------|------|------|
| **正向奖励（越大越好）** | | | |
| `teleop_joint_position` | 8.0 | 10.0 | 走路更强调关节轨迹跟踪 |
| `teleop_body_position_feet` | 12.0 | 8.0 | 站立更强调脚部稳定，走路允许移动 |
| `teleop_body_velocity_extend` | 3.0 | **6.0** | ⚡走路强调速度匹配 |
| `teleop_joint_velocity` | 2.0 | **5.0** | ⚡走路强调关节速度匹配 |
| `teleop_vr_3point` | 0.3 | 2.0 | 走路更注重上身协调 |
| **惩罚项（越小越好）** | | | |
| `penalty_action_rate` | -0.1 | **-0.005** | ⚡走路鼓励连续动作，几乎不惩罚 |
| `penalty_torques` | -0.0000001 | -0.0000005 | 走路允许更大扭矩 |
| `penalty_dof_acc` | 无 | -0.000001 | 走路轻微惩罚加速度 |
| `termination` | -200.0 | -300.0 | 走路更强调避免摔倒 |

### 3. 课程学习参数

| 参数 | 站立 | 走路 | 说明 |
|------|------|------|------|
| `reward_initial_penalty_scale` | 0.1 | 0.05 | 走路初始惩罚更低 |
| `reward_penalty_level_down_threshold` | 40 | 100 | 走路阈值更宽松 |
| `reward_penalty_level_up_threshold` | 42 | 300 | 走路阈值更宽松 |
| `reward_penalty_degree` | 0.00001 | 0.000005 | 走路变化速度更慢 |

---

## 🔑 关键差异总结

### 1️⃣ 速度相关 (最关键！)

**站立姿态**:
```yaml
teleop_body_vel: 0.2        # 极小sigma，强制速度≈0
teleop_body_ang_vel: 0.2    # 极小sigma，强制角速度≈0
teleop_joint_vel: 0.5       # 小sigma，要求关节静止
```
**结果**: 机器人**不敢动**，任何速度都会被严重惩罚

**走路动作**:
```yaml
teleop_body_vel: 1.5        # 大sigma，鼓励速度匹配
teleop_body_ang_vel: 1.5    # 大sigma，允许转动
teleop_joint_vel: 1.5       # 大sigma，鼓励关节运动
```
**结果**: 机器人**可以动**，跟随参考轨迹的速度

### 2️⃣ 动作连续性

**站立姿态**:
```yaml
penalty_action_rate: -0.1   # 惩罚动作变化
```
**结果**: 鼓励保持当前动作，不频繁调整

**走路动作**:
```yaml
penalty_action_rate: -0.005  # 几乎不惩罚
```
**结果**: 鼓励连续流畅的动作序列

### 3️⃣ 位置精确度要求

**站立姿态**:
- 容忍较大的位置偏差（sigma大）
- 但要求**绝对静止**

**走路动作**:
- 要求较高的轨迹跟踪精度（sigma小）
- 但允许**动态运动**

---

## 📝 实际表现对比

### 问题场景：机器人迈出一步就不敢动

**原因分析**（使用站立奖励函数训练走路）:

1. **速度被严重惩罚**
   - `teleop_body_vel: 0.2` → 任何速度都会大幅降低奖励
   - 机器人学到：保持静止 = 高奖励

2. **动作变化被惩罚**
   - `penalty_action_rate: -0.1` → 改变动作会被惩罚
   - 机器人学到：不要动 = 避免惩罚

3. **脚部位置奖励过高**
   - `teleop_body_position_feet: 12.0` + 小sigma
   - 机器人学到：脚不动 = 最大奖励

**解决方案**（使用走路奖励函数）:

1. **速度匹配成为正奖励**
   - `teleop_body_vel: 1.5` + 高权重 → 鼓励跟随参考速度
   - 机器人学到：匹配速度 = 高奖励

2. **连续动作不被惩罚**
   - `penalty_action_rate: -0.005` → 可以自由改变动作
   - 机器人学到：连续动作 = 允许的

3. **整体轨迹跟踪均衡**
   - 关节、速度、位置都很重要
   - 机器人学到：完整执行动作序列 = 最大奖励

---

## 🎯 使用建议

### 何时使用站立奖励函数？
✅ 训练静态姿态保持（如站立、蹲马步等）
✅ 需要机器人保持某个固定姿势
✅ 强调稳定性而非动作执行

### 何时使用走路奖励函数？
✅ 训练任何动态运动（走路、跑步、跳跃等）
✅ 需要执行连续的动作序列
✅ 强调轨迹跟踪和运动协调

### 训练脚本自动选择

运行 `./train_standing_from_cr7.sh`:
- **选项1 (站立姿态)** → 自动使用 `reward_motion_tracking_standing`
- **选项2-5 (走路L1-L4)** → 自动使用 `reward_motion_tracking_walking`

---

## 🔧 调优建议

### 如果走路训练不收敛

可以尝试调整 `reward_motion_tracking_walking.yaml` 中的以下参数：

1. **增加轨迹跟踪奖励**:
   ```yaml
   teleop_joint_position: 15.0  # 从10.0增加
   teleop_body_velocity_extend: 8.0  # 从6.0增加
   ```

2. **放宽Sigma容差**（降低难度）:
   ```yaml
   teleop_feet_pos: 0.12  # 从0.08放宽
   teleop_joint_pos: 2.0  # 从1.5放宽
   ```

3. **调整课程学习速度**:
   ```yaml
   reward_penalty_degree: 0.000002  # 更慢的课程进度
   ```

### 如果站立训练不稳定

可以尝试调整 `reward_motion_tracking_standing.yaml` 中的以下参数：

1. **进一步降低速度惩罚**:
   ```yaml
   teleop_body_vel: 0.1  # 从0.2降低
   teleop_joint_vel: 0.3  # 从0.5降低
   ```

2. **增加脚部位置奖励**:
   ```yaml
   teleop_body_position_feet: 15.0  # 从12.0增加
   ```

---

## 📚 参考

- 站立配置: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_standing.yaml`
- 走路配置: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_walking.yaml`
- 训练脚本: `train_standing_from_cr7.sh`
- 使用指南: `MOTION_TRAINING_GUIDE.md`

---

**最后更新**: 2025-10-23

