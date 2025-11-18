# Motion Tracking奖励调整指南

## 🎯 调整目标

将Motion Tracking的奖励函数从**精确模仿动态动作**调整为**鼓励稳定站立**。

---

## 📊 关键调整对比

### 1. 模仿奖励权重调整

| 奖励项 | 原值 | 新值 | 原因 |
|--------|------|------|------|
| `teleop_joint_position` | 0.75 | **1.5** ✅ | 增加关节跟踪重要性 |
| `teleop_body_position_feet` | 2.1 | **3.0** ✅ | 脚部位置最重要 |
| `teleop_body_rotation` | 0.5 | **1.0** ✅ | 保持直立很重要 |
| `teleop_velocity` | 0.5 | **0.8** ✅ | 站立时应静止 |
| `teleop_vr_3point` | 1.6 | **0.5** ⬇️ | 手和头不重要 |

### 2. 惩罚降低（关键改进）⭐

| 惩罚项 | 原值 | 新值 | 改进 |
|--------|------|------|------|
| `penalty_action_rate` | -0.5 | **-0.1** ✅ | 允许微调平衡 |
| `penalty_torques` | -1e-06 | **-5e-07** ✅ | 允许使用力量 |
| `penalty_feet_ori` | -2.0 | **-0.5** ✅ | 脚姿态不要太严格 |
| `penalty_slippage` | -1.0 | **-0.3** ✅ | 允许微小滑动 |
| `limits_dof_pos` | -10.0 | **-5.0** ✅ | 关节限制放宽 |
| `termination` | -200.0 | **-100.0** ✅ | 降低倒下惩罚 |

### 3. 容差参数放宽（核心改进）🌟

```yaml
# 原配置（隐式）
teleop_joint_position_sigma: ~0.2  # 小容差，要求精确匹配

# 新配置
teleop_joint_position_sigma: 0.4   # 大容差，允许更大偏差
teleop_gaussian_sigma: 0.3         # 整体放宽
```

**效果**：
- 原来：关节偏差0.2会导致奖励快速衰减
- 现在：关节偏差0.4仍能获得较高奖励

---

## 🔬 调整原理

### 问题诊断

之前的配置适合**动态动作模仿**：
```
目标：精确复现每一帧
要求：关节角度、速度、加速度都要接近
结果：对静态站立太严格 → 机器人不敢动 → 失去平衡 → 倒下
```

### 调整策略

新配置针对**静态站立**：
```
目标：保持稳定的站立姿态
允许：微调动作以保持平衡
鼓励：长时间站立、脚部稳定
结果：机器人可以主动调整 → 学会平衡
```

---

## 📈 预期改进

### 当前状态（原奖励）
```
Mean episode length: ~8步 (0.16秒)
Mean reward: 0.93 ~ 1.00
joint_pos_diff_norm: 0.58
rew_teleop_joint_position: 0.0059  ❌ 太低
```

### 预期效果（新奖励）
```
Mean episode length: > 30步 (0.6秒+)  ✅ 提升3-4倍
Mean reward: 2.0 ~ 4.0               ✅ 提升2-4倍
joint_pos_diff_norm: 0.6 ~ 0.8       ⚠️ 可能略增但可接受
rew_teleop_joint_position: > 0.5     ✅ 大幅提升
```

**关键**：允许偏差增加一点，但换来稳定性大幅提升！

---

## 🚀 使用方法

### 方式1: 使用训练脚本（推荐）

```bash
# 脚本已自动使用新奖励配置
./train_standing_from_cr7.sh

# 选择训练模式
# 建议先选择 2 (小规模验证) 看效果
```

### 方式2: 手动命令

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_standing \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=4096 \
  project_name=StandingOptimized \
  experiment_name=Standing_New_Rewards \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/default_standing.pkl" \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.8 \
  robot.asset.self_collisions=0 \
  algo.config.num_learning_iterations=500 \
  headless=True
```

---

## 📊 监控指标

训练时重点关注：

### 1. Episode长度（最重要）
```
目标: 从 8步 提升到 30-50步以上
指标: Env/average_episode_length
```

### 2. 总奖励
```
目标: 从 1.0 提升到 2.0-4.0
指标: Mean reward
```

### 3. 关节位置奖励
```
目标: 从 0.006 提升到 0.3-0.5
指标: rew_teleop_joint_position
```

### 4. 动作率惩罚（应该降低）
```
期望: 从 -0.003 增加到 -0.01（绝对值增大表示更多动作）
指标: rew_penalty_action_rate
说明: 这是好事！说明机器人在主动调整平衡
```

---

## 🔧 进一步调整

如果效果还是不理想，可以继续调整：

### 1. 进一步放宽惩罚
```yaml
# 在 reward_motion_tracking_standing.yaml 中
penalty_action_rate: -0.05  # 从-0.1改为-0.05
limits_dof_pos: -2.0        # 从-5.0改为-2.0
```

### 2. 增加站立时间奖励
```yaml
# 可以添加新的奖励项（需要在代码中实现）
standing_time_bonus: 0.1  # 每步额外奖励
```

### 3. 放宽终止条件
```bash
# 在训练命令中添加
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=1.5  # 从0.5改为1.5
```

---

## ⚠️ 注意事项

### 权衡

允许更大偏差 ↔ 提高稳定性

```
精确模仿（原配置）
   ↓
   严格要求 → 不敢动 → 倒下快
   
灵活站立（新配置）
   ↓
   允许调整 → 主动平衡 → 站得稳
```

### 何时回退

如果发现：
- 机器人动作过于随意
- 关节角度偏差 > 1.0
- 完全不像站立姿态

可以回退到原配置：
```bash
+rewards=motion_tracking/reward_motion_tracking_dm_2real
```

---

## 📚 配置文件

- **原配置**: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml`
- **新配置**: `humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_standing.yaml`
- **训练脚本**: `train_standing_from_cr7.sh`

---

## 🎓 核心思想

**Motion Tracking本质**：
```
min ||当前状态 - 参考状态||
```

**原配置**：严格的 L2 范数（精确匹配）
**新配置**：放宽的容差 + 稳定性加权

**类比**：
- 原配置 = "必须站成雕塑一样一动不动"
- 新配置 = "站稳就行，可以微调保持平衡"

---

**立即尝试新配置，看看效果！** 🚀

预期50-100次迭代内就能看到明显改善。

