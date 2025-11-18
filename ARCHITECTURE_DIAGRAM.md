# 🏗️ Standing Task - 架构图

## 📂 文件目录结构

```
ASAP/
├── humanoidverse/
│   ├── envs/
│   │   ├── base_task/
│   │   │   └── base_task.py (BaseTask)
│   │   ├── legged_base_task/
│   │   │   └── legged_robot_base.py (LeggedRobotBase) ← 共享基类
│   │   ├── motion_tracking/          ← 现有任务（不修改）
│   │   │   ├── __init__.py
│   │   │   └── motion_tracking.py
│   │   ├── locomotion/               ← 现有任务（不修改）
│   │   │   ├── __init__.py
│   │   │   └── locomotion.py
│   │   └── standing/                 ← ✨ 新增任务
│   │       ├── __init__.py           ← ✨ 新建
│   │       └── standing.py           ← ✨ 新建
│   │
│   ├── config/
│   │   ├── exp/
│   │   │   ├── motion_tracking.yaml  ← 现有（不修改）
│   │   │   ├── locomotion.yaml       ← 现有（不修改）
│   │   │   └── standing.yaml         ← ✨ 新建
│   │   │
│   │   ├── env/
│   │   │   ├── motion_tracking.yaml  ← 现有（不修改）
│   │   │   ├── locomotion.yaml       ← 现有（不修改）
│   │   │   └── standing.yaml         ← ✨ 新建
│   │   │
│   │   ├── obs/
│   │   │   ├── motion_tracking/      ← 现有（不修改）
│   │   │   ├── loco/                 ← 现有（不修改）
│   │   │   └── standing/             ← ✨ 新建目录
│   │   │       └── standing_obs_basic.yaml ← ✨ 新建
│   │   │
│   │   ├── rewards/
│   │   │   ├── motion_tracking/      ← 现有（不修改）
│   │   │   ├── loco/                 ← 现有（不修改）
│   │   │   └── standing/             ← ✨ 新建目录
│   │   │       └── reward_standing_g1.yaml ← ✨ 新建
│   │   │
│   │   ├── robot/                    ← 共享（不修改）
│   │   │   └── g1/g1_29dof_anneal_23dof.yaml
│   │   ├── terrain/                  ← 共享（不修改）
│   │   │   └── terrain_locomotion_plane.yaml
│   │   └── simulator/                ← 共享（不修改）
│   │       └── isaacgym.yaml
│   │
│   ├── agents/ppo/ppo.py             ← 共享（不修改）
│   ├── train_agent.py                ← 共享（不修改）
│   └── eval_agent.py                 ← 共享（不修改）
│
└── docs/                             ← ✨ 新建文档
    ├── STANDING_TASK_README.md       ← ✨ 完整教程
    ├── STANDING_VS_MOTION_TRACKING.md ← ✨ 对比分析
    ├── QUICK_START_STANDING.md       ← ✨ 快速开始
    ├── STANDING_TASK_SUMMARY.md      ← ✨ 实现总结
    └── ARCHITECTURE_DIAGRAM.md       ← ✨ 本文件
```

---

## 🔄 类继承关系

```
┌──────────────────┐
│    BaseTask      │ ← 最基础的任务类
└────────┬─────────┘
         │
         │ 继承
         ▼
┌──────────────────┐
│ LeggedRobotBase  │ ← 腿式机器人通用基类
└────────┬─────────┘   - 提供通用奖励函数
         │             - 提供通用观测函数
         │             - 提供reset/step逻辑
         │
         ├─────────────────────┬──────────────────┐
         │                     │                  │
         ▼                     ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ MotionTracking   │  │   Locomotion     │  │   Standing       │
│    任务类        │  │     任务类       │  │    任务类 ✨     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
  跟踪复杂动作           行走导航              稳定站立
  (现有，不修改)         (现有，不修改)         (新增)
```

---

## 🎯 Standing任务数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练开始                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 加载配置文件 (Hydra)                                         │
│     - exp/standing.yaml                                         │
│     - env/standing.yaml                                         │
│     - rewards/standing/reward_standing_g1.yaml                  │
│     - obs/standing/standing_obs_basic.yaml                      │
│     - robot/g1/g1_29dof_anneal_23dof.yaml                       │
│     - terrain/terrain_locomotion_plane.yaml                     │
│     - simulator/isaacgym.yaml                                   │
│     - algo/ppo.yaml                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 初始化环境 (LeggedRobotStanding)                             │
│     - 创建4096个并行仿真环境                                     │
│     - 加载G1机器人模型                                           │
│     - 初始化地形（平面）                                         │
│     - 设置站立目标：height=0.78m, feet_distance=0.30m           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 初始化PPO算法                                                │
│     - Actor网络 (80维输入 → 23维输出)                           │
│     - Critic网络 (80维输入 → 1维价值)                           │
│     - 优化器 (Adam)                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 训练循环 (每个iteration)                                     │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ 4.1 收集经验 (Rollout)                                   │ │
│     │     For step in range(num_steps_per_env):               │ │
│     │       - 获取观测 (80维)                                  │ │
│     │         ├─ dof_pos (23)                                 │ │
│     │         ├─ dof_vel (23)                                 │ │
│     │         ├─ actions (23)                                 │ │
│     │         ├─ base_lin_vel (3)                             │ │
│     │         ├─ base_ang_vel (3)                             │ │
│     │         ├─ projected_gravity (3)                        │ │
│     │         ├─ target_height (1)                            │ │
│     │         └─ feet_distance (1)                            │ │
│     │       - Actor推理 → 动作 (23维)                          │ │
│     │       - 仿真step (50Hz)                                 │ │
│     │       - 计算奖励 (12个奖励项)                            │ │
│     │       - 检查终止条件 (4个)                               │ │
│     │       - 存储到buffer                                    │ │
│     └─────────────────────────────────────────────────────────┘ │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ 4.2 更新策略 (PPO Update)                                │ │
│     │     For epoch in range(num_learning_epochs):            │ │
│     │       For minibatch in minibatches:                     │ │
│     │         - 计算advantage                                 │ │
│     │         - 计算actor loss (PPO clip)                     │ │
│     │         - 计算critic loss (MSE)                         │ │
│     │         - 反向传播                                      │ │
│     │         - 更新参数                                      │ │
│     └─────────────────────────────────────────────────────────┘ │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ 4.3 记录日志                                             │ │
│     │     - TensorBoard: reward, episode_length, losses       │ │
│     │     - 控制台: FPS, mean_reward, episode_length          │ │
│     │     - 保存checkpoint (每save_interval步)                 │ │
│     └─────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 训练完成                                                     │
│     - 保存最终模型                                               │
│     - 日志存储在 logs/Standing/<timestamp>-<experiment_name>/   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💰 奖励计算流程

```
每个仿真step (50Hz):

┌─────────────────────────────────────────────────────────────────┐
│  观测当前状态                                                    │
│  ├─ base_height                                                 │
│  ├─ projected_gravity                                           │
│  ├─ base_lin_vel, base_ang_vel                                  │
│  ├─ feet_positions                                              │
│  ├─ dof_pos, dof_vel                                            │
│  └─ contact_forces                                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  计算7个正向奖励 (reward_scales > 0)                             │
│  ├─ standing_upright (2.0)                                      │
│  │   └─ exp(-(projected_gravity[:2]**2).sum() / 0.1)           │
│  ├─ standing_base_height (1.5)                                  │
│  │   └─ exp(-(height - 0.78)**2 / 0.02)                        │
│  ├─ standing_feet_placement (1.0)                               │
│  │   └─ exp(-(feet_distance - 0.30)**2 / 0.05)                 │
│  ├─ standing_base_stability (1.0)                               │
│  │   └─ exp(-base_pos_xy.sum() / 0.01)                         │
│  ├─ standing_zero_velocity (0.8)                                │
│  │   └─ [exp(-lin_vel**2/0.05) + exp(-ang_vel**2/0.05)] / 2    │
│  ├─ standing_symmetric_stance (0.5)                             │
│  │   └─ exp(-(height_diff + y_diff) / 0.05)                    │
│  └─ standing_joint_default (0.3)                                │
│      └─ exp(-(dof_pos - default)**2 / 0.5)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  计算12个惩罚项 (reward_scales < 0)                              │
│  ├─ penalty_orientation (-2.0)                                  │
│  ├─ penalty_base_height_deviation (-2.0)                        │
│  ├─ penalty_lin_vel (-1.5)                                      │
│  ├─ penalty_ang_vel (-1.5)                                      │
│  ├─ penalty_feet_ori (-1.0)                                     │
│  ├─ penalty_uneven_feet_contact (-0.8)                          │
│  ├─ penalty_slippage (-0.5)                                     │
│  ├─ penalty_action_rate (-0.05)                                 │
│  ├─ penalty_torques (-0.00002)                                  │
│  ├─ limits_dof_pos (-10.0)                                      │
│  ├─ limits_dof_vel (-5.0)                                       │
│  └─ limits_torque (-5.0)                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  总奖励 = Σ(reward_i * scale_i * dt)                            │
│  其中 dt = 0.02 (50Hz)                                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  检查终止条件                                                    │
│  ├─ 摔倒: height < 0.50m                                        │
│  ├─ 跳跃: height > 1.20m                                        │
│  ├─ 倾斜: tilt > 0.5rad                                         │
│  ├─ 重力: gravity_xy > 0.7                                      │
│  └─ 超时: episode_length > 20s                                  │
│  如果终止: reward += termination (-200.0 * dt)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔀 与Motion Tracking的对比流程

### Motion Tracking 数据流
```
训练step:
┌────────────────────────────────────┐
│ 1. 从Motion Library加载参考动作    │
│    motion_lib.get_motion_state()   │
│    → ref_body_pos (24个刚体)       │
│    → ref_body_rot (24个刚体)       │
│    → ref_joint_pos (23个关节)      │
│    → ref_joint_vel (23个关节)      │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 2. 计算与参考动作的差异            │
│    dif_pos = ref_pos - curr_pos    │
│    dif_rot = ref_rot * inv(curr)   │
│    dif_vel = ref_vel - curr_vel    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 3. 计算8个跟踪奖励                 │
│    基于差异的高斯奖励              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 4. 检查是否偏离太远                │
│    if norm(dif_pos) > threshold:   │
│       terminate = True             │
└────────────────────────────────────┘
```

### Standing 数据流
```
训练step:
┌────────────────────────────────────┐
│ 1. 固定的站立目标（无需加载文件）  │
│    target_height = 0.78m           │
│    target_feet_distance = 0.30m    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 2. 计算与目标的差异                │
│    height_error = |curr - 0.78|    │
│    feet_error = |dist - 0.30|      │
│    tilt_error = projected_gravity  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 3. 计算7个站立奖励                 │
│    基于固定目标的高斯奖励          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 4. 检查是否摔倒                    │
│    if height < 0.50 or tilt > 0.5: │
│       terminate = True             │
└────────────────────────────────────┘
```

---

## 📊 配置文件依赖图

```
train_agent.py
      │
      ├─→ +exp=standing
      │     └─→ config/exp/standing.yaml
      │           ├─→ defaults: /algo: ppo
      │           └─→ defaults: /env: standing
      │
      ├─→ +rewards=standing/reward_standing_g1
      │     └─→ config/rewards/standing/reward_standing_g1.yaml
      │           └─→ 定义所有reward_scales
      │
      ├─→ +obs=standing/standing_obs_basic
      │     └─→ config/obs/standing/standing_obs_basic.yaml
      │           └─→ 定义obs_dict, obs_dims, obs_scales
      │
      ├─→ +robot=g1/g1_29dof_anneal_23dof
      │     └─→ config/robot/g1/g1_29dof_anneal_23dof.yaml
      │           └─→ 定义机器人参数（共享，不修改）
      │
      ├─→ +terrain=terrain_locomotion_plane
      │     └─→ config/terrain/terrain_locomotion_plane.yaml
      │           └─→ 定义地形（共享，不修改）
      │
      └─→ +simulator=isaacgym
            └─→ config/simulator/isaacgym.yaml
                  └─→ 定义仿真器参数（共享，不修改）
```

---

## 🎛️ 运行时架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python进程 (train_agent.py)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Hydra配置管理                                 │  │
│  │  - 加载所有YAML配置                                        │  │
│  │  - 处理overrides                                          │  │
│  │  - 生成最终config对象                                      │  │
│  └────────────────────┬──────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         LeggedRobotStanding (环境)                        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Isaac Gym Simulator (GPU加速)                      │  │  │
│  │  │  - 4096个并行环境                                   │  │  │
│  │  │  - 每个环境独立仿真                                 │  │  │
│  │  │  - 200Hz物理仿真 → 50Hz控制                        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  奖励计算 (12个函数) ────────────┐                         │  │
│  │  观测生成 (8个函数) ──────────┐  │                         │  │
│  │  终止检查 (4个条件) ────────┐ │  │                         │  │
│  └────────────┬────────────────┼─┼──┼─────────────────────────┘  │
│               │                │ │  │                             │
│               │ observations   │ │  │ rewards                     │
│               │ (80维 × 4096)  │ │  │ (1维 × 4096)                │
│               │                │ │  │                             │
│               ▼                │ │  ▼                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               PPO算法 (agents/ppo/ppo.py)                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Actor Network (MLP)                                │  │  │
│  │  │  Input: 80维观测                                    │  │  │
│  │  │  Hidden: [512, 256, 128]                            │  │  │
│  │  │  Output: 23维动作 (高斯分布的均值)                  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Critic Network (MLP)                               │  │  │
│  │  │  Input: 80维观测                                    │  │  │
│  │  │  Hidden: [512, 256, 128]                            │  │  │
│  │  │  Output: 1维价值函数                                │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  优化器 (Adam) ──────────────────────────────────────────┐ │  │
│  └────────────┬────────────────────────────────────────────┼─┘  │
│               │                                            │     │
│               │ actions (23维 × 4096)                      │     │
│               │                                            │     │
│               └───────────────┐                            │     │
│                               │                            │     │
│                               ▼                            │     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │             TensorBoard / Logging                         │  │
│  │  - Train/mean_reward                                      │  │
│  │  - Train/mean_episode_length                              │  │
│  │  - Loss/Surrogate, Loss/Value                             │  │
│  │  - Env/standing_upright, Env/standing_base_height         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    logs/Standing/<run_dir>/
                    ├── model_1000.pt
                    ├── model_2000.pt
                    ├── ...
                    ├── config.yaml
                    └── events.out.tfevents.*
```

---

## ✅ 总结

这个架构图展示了：

1. **文件组织** - Standing任务的所有新文件都在独立目录
2. **类继承** - Standing继承自LeggedRobotBase，复用基础功能
3. **数据流** - 从配置加载到训练循环的完整流程
4. **奖励计算** - 12个奖励项的详细计算逻辑
5. **对比** - 与Motion Tracking的核心差异
6. **配置依赖** - 所有YAML文件的依赖关系
7. **运行时** - Python进程、GPU仿真、神经网络的交互

**核心原则：独立性 + 可扩展性 + 不影响现有代码** ✨

