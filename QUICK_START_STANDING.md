# 🚀 Standing Task - 快速开始

## ⚡ 一键训练

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=standing \
+rewards=standing/reward_standing_g1 \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=standing/standing_obs_basic \
num_envs=4096 project_name=Standing \
experiment_name=My_Standing_Task
```

## 📊 监控训练

```bash
tensorboard --logdir=logs/Standing
```

访问：http://localhost:6006

## 🎮 评估模型

```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/Standing/<your_run_dir>/model_5000.pt
```

## 🎯 核心奖励（站立任务）

| 奖励 | 权重 | 目标 |
|-----|------|------|
| `standing_upright` | 2.0 | 保持垂直 ⭐ |
| `standing_base_height` | 1.5 | 高度0.78m |
| `standing_feet_placement` | 1.0 | 双脚间距0.30m |
| `standing_zero_velocity` | 0.8 | 保持静止 |

## 🔧 调优快速参考

### 容易摔倒？
```yaml
# 在 reward_standing_g1.yaml 中修改
standing_upright: 3.0  # 增大权重
penalty_orientation: -3.0
```

### 站立不稳（晃动）？
```yaml
standing_zero_velocity: 1.5  # 增大权重
penalty_lin_vel: -2.0
penalty_ang_vel: -2.0
```

### 双脚距离不合理？
```yaml
desired_feet_distance: 0.25  # 调整目标距离
standing_feet_placement: 1.5  # 增大权重
```

## ✅ 成功标准

- 平均回合长度 > 400步
- 平均奖励 > 15.0
- 身高误差 < 3cm
- 姿态倾斜 < 5度

## 📁 新增文件清单

```
config/
├── exp/standing.yaml
├── env/standing.yaml
├── obs/standing/standing_obs_basic.yaml
└── rewards/standing/reward_standing_g1.yaml

envs/standing/
├── __init__.py
└── standing.py
```

**✅ 不影响Motion Tracking任务！**

## 📚 详细文档

- 📖 完整说明：`STANDING_TASK_README.md`
- 🔍 对比分析：`STANDING_VS_MOTION_TRACKING.md`
- 🧪 测试脚本：`./test_standing_task.sh`

