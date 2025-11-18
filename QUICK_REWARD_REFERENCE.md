# 奖励函数快速参考卡

## 🚨 问题诊断

### 症状：机器人迈出一步就不敢动
**原因**: 使用了站立奖励函数训练走路动作
**解决**: 使用走路奖励函数

---

## ⚡ 关键差异（3个最重要的参数）

| 参数 | 站立 | 走路 | 影响 |
|------|------|------|------|
| `teleop_body_vel` sigma | **0.2** | **1.5** | 速度是否被惩罚 |
| `penalty_action_rate` | **-0.1** | **-0.005** | 动作连续性 |
| `teleop_body_velocity_extend` | 3.0 | **6.0** | 速度匹配奖励 |

---

## 📋 完整对比表

### Sigma值 (越小=要求越严格)

```
参数                    站立    走路    解释
──────────────────────────────────────────────
速度相关（关键！）:
  body_vel              0.2 →  1.5    🔥 走路鼓励速度
  body_ang_vel          0.2 →  1.5    🔥 走路鼓励转动
  joint_vel             0.5 →  1.5    🔥 走路鼓励关节运动

位置相关:
  upper_body_pos        0.2 →  0.08   走路要求更精确
  lower_body_pos        0.5 →  0.12   走路要求更精确
  feet_pos              0.15 → 0.08   走路要求更精确
  joint_pos             3.0 →  1.5    走路要求更精确
```

### 奖励权重

```
奖励项                         站立    走路    说明
──────────────────────────────────────────────────────
正向奖励:
  teleop_joint_position        8.0 →  10.0   走路更重视轨迹
  teleop_body_velocity_extend  3.0 →  6.0    🔥 走路强调速度
  teleop_joint_velocity        2.0 →  5.0    🔥 走路强调关节速度
  teleop_body_position_feet    12.0 → 8.0    站立强调脚稳定
  teleop_vr_3point             0.3 →  2.0    走路需要上身协调

惩罚项:
  penalty_action_rate          -0.1 → -0.005  🔥 走路几乎不惩罚
  penalty_torques              -1e-7→ -5e-7   走路允许更大扭矩
  termination                  -200 → -300    走路更严格
```

---

## 🎯 训练策略总结

### 站立姿态
```
目标: 保持静止
策略: 惩罚任何速度和动作变化
结果: 机器人学会"不动"
```

### 走路动作
```
目标: 跟随运动轨迹
策略: 奖励速度匹配，允许连续动作
结果: 机器人学会"跟着走"
```

---

## 💡 使用方法

### 自动选择（推荐）
```bash
./train_standing_from_cr7.sh

选择1: 站立 → 自动用站立奖励
选择2-5: 走路 → 自动用走路奖励
```

### 手动指定
```bash
# 站立
python train_agent.py +rewards=motion_tracking/reward_motion_tracking_standing

# 走路
python train_agent.py +rewards=motion_tracking/reward_motion_tracking_walking
```

---

## 🔧 常见调优

### 走路不收敛？

1. **降低难度** (增加sigma):
   ```yaml
   teleop_feet_pos: 0.12  # 从0.08放宽
   ```

2. **增加奖励**:
   ```yaml
   teleop_joint_position: 15.0  # 从10.0增加
   ```

3. **减慢课程学习**:
   ```yaml
   reward_penalty_degree: 0.000002  # 更慢
   ```

### 站立不稳？

1. **更严格要求静止**:
   ```yaml
   teleop_body_vel: 0.1  # 从0.2降低
   ```

2. **增加脚部奖励**:
   ```yaml
   teleop_body_position_feet: 15.0  # 从12.0增加
   ```

---

## 📊 预期训练效果

### 站立姿态
- 收敛迭代: **100-500**
- 主要指标: 位置误差 < 0.1
- 成功标志: 不倒下、不晃动

### 走路动作
- 收敛迭代: **1000-2000**
- 主要指标: 轨迹跟踪误差逐渐降低
- 成功标志: 完整执行走路序列

---

## 📁 配置文件位置

```
humanoidverse/config/rewards/motion_tracking/
├── reward_motion_tracking_standing.yaml  ← 站立
└── reward_motion_tracking_walking.yaml   ← 走路
```

---

**提示**: 详细对比请参阅 `REWARD_COMPARISON.md`

