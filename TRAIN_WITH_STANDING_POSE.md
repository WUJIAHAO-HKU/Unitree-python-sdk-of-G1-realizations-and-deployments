# 使用提取的站立姿态进行训练

## 📋 概述

我们已经成功从 CR7 动作的 2.8 秒时刻提取了站立姿态数据。现在可以使用这个姿态训练机器人保持站立。

## 📊 提取的数据

**文件位置**: `humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl`

**数据内容**:
- ✅ **根位置** (root_trans_offset): `[-0.131, -0.688, 0.848]` (x, y, z 米)
- ✅ **根旋转** (root_rot): 四元数 `[0.023, 0.013, 0.700, 0.714]` (x, y, z, w)
- ✅ **关节角度** (dof): 23个关节的角度值
- ✅ **姿态** (pose_aa): 27个关节的轴角表示 (27, 3)
- ✅ **速度数据**: 所有速度设为0（站立状态）
  - root_lin_vel = [0, 0, 0]
  - root_ang_vel = [0, 0, 0]
  - dof_vel = zeros(23)

## 🎯 两种使用方式

### 方式1: Motion Tracking 任务（推荐）

使用 Motion Tracking 任务让机器人学习保持这个站立姿态：

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=4096 \
  project_name=StandingFromCR7 \
  experiment_name=Standing_MotionTracking \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_penalty_degree=0.00001 \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=True \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.2 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  robot.asset.self_collisions=0 \
  headless=True
```

**优点**:
- ✅ 使用成熟的 Motion Tracking 奖励函数
- ✅ 自动跟踪提取的真实站立姿态
- ✅ 有课程学习机制
- ✅ 可以精确复现 CR7 的站立姿态

**原理**:
- 机器人会尝试匹配提取的站立姿态的所有状态（位置、旋转、关节角度）
- 由于只有一帧，机器人会学习保持这个姿态不动
- 速度都是0，所以机器人学会静止站立

---

### 方式2: 自定义 Standing 任务

使用我们之前创建的 Standing 任务，但参考提取的关节角度：

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=standing \
  +domain_rand=NO_domain_rand \
  +rewards=standing/reward_standing_g1 \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=standing/standing_obs_basic \
  num_envs=4096 \
  project_name=StandingTask \
  experiment_name=Standing_CustomRewards \
  robot.asset.self_collisions=0 \
  headless=True
```

**优点**:
- ✅ 可以自定义奖励权重
- ✅ 更灵活的站立策略（不必完全复现 CR7 姿态）
- ✅ 可以调整目标高度和脚距等参数

**注意**: 这种方式不直接使用提取的 `.pkl` 文件，而是使用奖励函数引导机器人学习站立。

---

## 🔍 对比两种方式

| 特性 | Motion Tracking | 自定义 Standing 任务 |
|------|----------------|---------------------|
| **使用提取的姿态** | ✅ 直接使用 | ❌ 不直接使用 |
| **姿态精确度** | 🎯 精确复现 CR7 姿态 | 📐 学习通用站立姿态 |
| **训练难度** | 中等 | 较简单 |
| **灵活性** | 低（固定姿态） | 高（可调整参数） |
| **适用场景** | 需要特定姿态 | 需要稳定站立 |

---

## 📈 训练监控

训练时关注以下指标：

### Motion Tracking 方式
```
rew_teleop_body_position_extend  # 身体位置跟踪
rew_teleop_body_rotation         # 身体旋转跟踪
rew_teleop_joint_position        # 关节角度跟踪
rew_teleop_joint_velocity        # 关节速度（应接近0）
rew_penalty_action_rate          # 动作平滑度
rew_termination                  # 终止惩罚
```

### Standing 任务方式
```
rew_standing_upright             # 保持直立
rew_standing_base_height         # 目标高度（0.78m）
rew_standing_feet_placement      # 脚间距（0.30m）
rew_standing_base_stability      # 基座稳定性
rew_standing_zero_velocity       # 零速度
```

---

## 🧪 快速测试

### 1. 测试提取的姿态文件是否有效

```bash
# 用1个环境快速测试
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=1 \
  project_name=TEST \
  experiment_name=TEST_StandingPose \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl" \
  algo.num_learning_iterations=1 \
  algo.num_steps_per_env=10 \
  robot.asset.self_collisions=0 \
  headless=False
```

如果能看到机器人尝试保持站立姿态，说明文件格式正确！

### 2. 可视化站立姿态

```bash
python extract_standing_pose.py \
  --motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl \
  --time 0.0 \
  --output /tmp/test.pkl \
  --visualize
```

---

## 📝 修改提取时刻

如果想从不同时刻提取站立姿态：

```bash
# 例如从 3.5 秒提取
python extract_standing_pose.py \
  --motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
  --time 3.5 \
  --output humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_3.5s.pkl \
  --visualize
```

---

## 🎓 推荐训练流程

### 阶段1: 验证姿态 (5-10分钟)
```bash
# 小规模测试，确保姿态正确
num_envs=256
algo.num_learning_iterations=50
```

### 阶段2: 初步训练 (1-2小时)
```bash
# 中等规模，观察收敛趋势
num_envs=1024
algo.num_learning_iterations=500
```

### 阶段3: 完整训练 (4-8小时)
```bash
# 大规模训练至收敛
num_envs=4096
algo.num_learning_iterations=2000
```

---

## ⚠️ 常见问题

### Q1: 机器人无法保持站立，直接倒下
**可能原因**:
- 提取的姿态本身不稳定（可以尝试其他时刻）
- 重力或物理参数不匹配
- 需要启用 `self_collisions`

**解决方法**:
```bash
robot.asset.self_collisions=1  # 启用自碰撞检测
env.config.termination.terminate_by_low_height=True  # 提早终止无效尝试
```

### Q2: 训练很慢，奖励不收敛
**可能原因**:
- 奖励权重不合理
- 课程学习参数太激进

**解决方法**:
```bash
# 放松终止条件
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.5

# 降低课程学习速度
rewards.reward_penalty_degree=0.000005
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.00001
```

### Q3: 想要调整站立高度
**如果使用 Motion Tracking**:
- 需要重新提取姿态，或手动修改 `.pkl` 文件中的 `root_trans_offset[2]`

**如果使用 Standing 任务**:
```bash
env.config.target_base_height=0.85  # 调整目标高度（米）
```

---

## 🚀 下一步

1. **训练模型**: 选择一种方式开始训练
2. **监控指标**: 使用 TensorBoard 观察训练进度
3. **评估模型**: 使用 `eval_agent.py` 测试训练好的模型
4. **调整参数**: 根据表现调整奖励权重或终止条件

---

## 📚 相关文档

- `EXTRACT_STANDING_POSE_GUIDE.md` - 姿态提取详细指南
- `STANDING_TASK_README.md` - Standing 任务完整文档
- `STANDING_VS_MOTION_TRACKING.md` - 两种任务的详细对比

---

**祝训练顺利！** 🎉

如有问题，请检查日志文件或使用 `headless=False` 可视化查看机器人行为。

