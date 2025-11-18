# Standing Task - 机器人站立平衡任务

## 📋 任务概述

这是一个全新的**站立平衡（Standing Balance）**任务，训练G1人形机器人保持稳定站立。

**与Motion Tracking的区别**：
- ❌ **不**跟踪参考动作
- ✅ **只**训练站立这一个简单但重要的技能
- ✅ 完全独立的配置文件，不影响现有训练

---

## 🎯 训练目标

1. **保持直立姿态**：身体垂直，不倾斜
2. **保持目标高度**：身高维持在 0.78m
3. **双脚合理间距**：双脚距离保持在 0.30m
4. **全身静止**：线速度和角速度接近零
5. **重心稳定**：质心不漂移

---

## 📂 新增文件清单

```
humanoidverse/
├── config/
│   ├── exp/
│   │   └── standing.yaml                          # 实验配置
│   ├── env/
│   │   └── standing.yaml                          # 环境配置
│   ├── obs/
│   │   └── standing/
│   │       └── standing_obs_basic.yaml            # 观测配置
│   └── rewards/
│       └── standing/
│           └── reward_standing_g1.yaml            # 奖励配置
└── envs/
    └── standing/
        ├── __init__.py
        └── standing.py                             # 核心任务实现
```

**✅ 保证：所有新文件都在独立目录/命名空间，不会影响现有的Motion Tracking任务！**

---

## 🚀 训练命令

### 基础训练
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
project_name=Standing \
experiment_name=Standing_G1_Basic \
robot.asset.self_collisions=0
```

### 多GPU训练（如果有多GPU）
```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=standing \
+domain_rand=NO_domain_rand \
+rewards=standing/reward_standing_g1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=standing/standing_obs_basic \
num_envs=8192 \
project_name=Standing \
experiment_name=Standing_G1_Large
```

---

## 🎮 评估命令

```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/Standing/<run_dir>/model_5000.pt
```

或者重新指定配置：
```bash
python humanoidverse/eval_agent.py \
+simulator=isaacgym \
+exp=standing \
+rewards=standing/reward_standing_g1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=standing/standing_obs_basic \
num_envs=1 \
eval_name=StandingEval \
+checkpoint=logs/Standing/<run_dir>/model_5000.pt
```

---

## 💰 奖励机制详解

### ✅ 正向奖励（鼓励站立）

| 奖励项 | 权重 | 说明 |
|-------|------|------|
| `standing_upright` | 2.0 | **最重要**：保持身体垂直 |
| `standing_base_height` | 1.5 | 保持目标高度 0.78m |
| `standing_feet_placement` | 1.0 | 双脚间距保持 0.30m |
| `standing_base_stability` | 1.0 | 质心稳定不漂移 |
| `standing_zero_velocity` | 0.8 | 保持静止 |
| `standing_symmetric_stance` | 0.5 | 对称站姿（左右对称）|
| `standing_joint_default` | 0.3 | 关节角度接近默认值 |

### ❌ 惩罚项（约束行为）

| 惩罚项 | 权重 | 说明 |
|-------|------|------|
| `penalty_orientation` | -2.0 | 惩罚身体倾斜 |
| `penalty_base_height_deviation` | -2.0 | 惩罚高度偏离 |
| `penalty_lin_vel` | -1.5 | 惩罚线速度（要求静止）|
| `penalty_ang_vel` | -1.5 | 惩罚角速度（要求静止）|
| `penalty_feet_ori` | -1.0 | 惩罚脚部姿态异常 |
| `penalty_uneven_feet_contact` | -0.8 | 惩罚双脚接触力不均 |
| `penalty_slippage` | -0.5 | 惩罚脚滑 |
| `penalty_action_rate` | -0.05 | 惩罚动作变化率 |
| `penalty_torques` | -0.00002 | 惩罚大力矩 |
| `limits_dof_pos` | -10.0 | 关节角度超限 |
| `limits_dof_vel` | -5.0 | 关节速度超限 |
| `limits_torque` | -5.0 | 力矩超限 |
| `termination` | -200.0 | 终止惩罚（摔倒）|

---

## 🔧 终止条件

机器人在以下情况会被重置：

1. **高度过低**：身高 < 0.50m（摔倒）
2. **高度过高**：身高 > 1.20m（异常跳跃）
3. **倾斜过大**：身体倾斜 > 28度（失去平衡）
4. **重力异常**：投影重力x或y分量 > 0.7（即将摔倒）
5. **超时**：20秒仍未稳定（训练时）

---

## 📊 观测空间

机器人通过以下传感器感知自身状态（总维度：80）：

| 观测类型 | 维度 | 说明 |
|---------|------|------|
| `obs_dof_pos` | 23 | 关节位置 |
| `obs_dof_vel` | 23 | 关节速度 |
| `obs_actions` | 23 | 上一步动作 |
| `obs_base_lin_vel` | 3 | 基座线速度（局部坐标系）|
| `obs_base_ang_vel` | 3 | 基座角速度（局部坐标系）|
| `obs_projected_gravity` | 3 | 投影重力（IMU）|
| `obs_standing_target_height` | 1 | 目标高度 |
| `obs_feet_distance` | 1 | 当前双脚距离 |

**Actor观测 = Critic观测 = 80维**（站立任务不需要特权信息）

---

## 🎛️ 关键参数调优建议

### 如果机器人容易摔倒
- 增大 `standing_upright` 权重：`2.0 → 3.0`
- 增大 `penalty_orientation` 权重：`-2.0 → -3.0`
- 降低 `termination_max_base_tilt`：`0.5 → 0.4`（更早终止）

### 如果站姿不稳定（晃动）
- 增大 `standing_zero_velocity` 权重：`0.8 → 1.5`
- 增大 `penalty_lin_vel` 和 `penalty_ang_vel`：`-1.5 → -2.0`
- 增大 `standing_base_stability` 权重：`1.0 → 1.5`

### 如果双脚距离不合适
- 调整 `desired_feet_distance`：默认 `0.30m`
- 增大 `standing_feet_placement` 权重：`1.0 → 1.5`

### 如果高度控制不准
- 增大 `standing_base_height` 权重：`1.5 → 2.0`
- 减小 `reward_tracking_sigma.base_height`：`0.02 → 0.01`（更严格）

---

## 🔄 与Motion Tracking对比

| 特性 | Motion Tracking | Standing |
|------|----------------|----------|
| **训练目标** | 跟踪复杂参考动作（CR7动作）| 保持稳定站立 |
| **难度** | 高（多维度跟踪）| 低（单一姿态）|
| **训练时长** | 数小时~数天 | 数十分钟~数小时 |
| **奖励维度** | 8个跟踪维度 + 约束 | 5个站立维度 + 约束 |
| **观测维度** | 数百维（含历史）| 80维（无历史）|
| **应用场景** | 全身动作模仿 | 站立平衡、初始化姿态 |
| **是否独立** | ✅ 完全独立 | ✅ 完全独立 |

---

## 🧪 验证成功标准

训练成功的指标：

1. **平均回合长度** > 400步（20秒 × 50Hz）
2. **平均奖励** > 15.0
3. **终止率** < 5%
4. **身高误差** < 3cm
5. **姿态倾斜** < 5度
6. **速度** < 0.05 m/s 和 rad/s

---

## 🐛 常见问题

### Q1: 训练时机器人一直摔倒？
**A**: 检查初始姿态是否合理：
```yaml
# 在 robot config 中确认 init_state.default_joint_angles
# 应该是稳定的站立姿态
```

### Q2: 奖励一直是负数？
**A**: 可能惩罚权重过大，尝试：
- 减小惩罚项权重（如 `penalty_*`）
- 增大正向奖励权重（如 `standing_*`）

### Q3: 机器人站立但一直晃动？
**A**: 增强速度约束：
```yaml
standing_zero_velocity: 1.5  # 原来 0.8
penalty_lin_vel: -2.0        # 原来 -1.5
penalty_ang_vel: -2.0        # 原来 -1.5
```

### Q4: 双脚距离不合理（太宽或太窄）？
**A**: 调整目标距离和权重：
```yaml
desired_feet_distance: 0.25  # 根据实际调整
standing_feet_placement: 1.5  # 增大权重
```

---

## 📈 训练进度监控

关键指标（TensorBoard）：

- `Train/mean_reward`：应该持续上升
- `Train/mean_episode_length`：应该接近最大值（400步）
- `Env/upper_body_diff_norm`：无（站立任务不跟踪）
- `Env/standing_upright`：应该接近1.0
- `Env/standing_base_height`：应该接近1.0
- `Loss/Surrogate`：应该逐渐收敛
- `Loss/Value`：应该逐渐减小

---

## 🎓 进阶：添加扰动测试

评估时可以添加外力扰动测试鲁棒性：

```python
# 在 domain_rand 配置中
push_robots: True
push_interval_s: [5, 10]  # 每5-10秒推一次
max_push_vel_xy: 0.5      # 最大推力 0.5 m/s
```

---

## 📝 总结

你现在有了：
- ✅ 完整的Standing任务实现
- ✅ 独立的配置文件系统
- ✅ 详细的奖励机制
- ✅ 清晰的训练/评估命令
- ✅ 不影响Motion Tracking任务

**开始训练吧！** 🚀

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=standing \
+rewards=standing/reward_standing_g1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=standing/standing_obs_basic \
num_envs=4096 project_name=Standing \
experiment_name=My_First_Standing
```

