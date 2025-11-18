# 站立姿态提取与训练 - 完整总结

## 🎯 任务完成情况

✅ **已完成**: 从 CR7 动作文件的 2.8 秒时刻成功提取站立姿态

## 📁 生成的文件

### 1. 核心文件
- **`extract_standing_pose.py`** - 姿态提取脚本
- **`humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl`** - 提取的站立姿态数据

### 2. 训练脚本
- **`train_standing_from_cr7.sh`** - 一键训练脚本（推荐使用）

### 3. 文档
- **`EXTRACT_STANDING_POSE_GUIDE.md`** - 姿态提取详细指南
- **`TRAIN_WITH_STANDING_POSE.md`** - 训练方法和参数说明
- **`STANDING_POSE_EXTRACTION_SUMMARY.md`** - 本文件（总结）

## 📊 提取的站立姿态数据

从 `0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl` 提取：

```
时刻: 2.8秒 (第84帧, fps=30)

根位置 (root_trans_offset):
  x: -0.131m
  y: -0.688m
  z:  0.848m  ← 站立高度

根旋转 (root_rot, 四元数 xyzw):
  [0.023, 0.013, 0.700, 0.714]

关节角度 (dof):
  23个关节的角度值
  前5个: [-0.146, 0.297, 0.586, 0.543, 0.024]

速度 (所有为0，表示静止站立):
  root_lin_vel: [0, 0, 0]
  root_ang_vel: [0, 0, 0]
  dof_vel: zeros(23)
```

## 🚀 快速开始

### 方式 1: 使用一键脚本（推荐）

```bash
# 运行脚本，按提示选择训练模式
./train_standing_from_cr7.sh
```

脚本提供三种模式：
1. **快速测试** - 1个环境，可视化，验证姿态是否正确
2. **小规模验证** - 256个环境，100次迭代，验证训练可行性
3. **完整训练** - 4096个环境，2000次迭代，完整训练

### 方式 2: 手动命令

```bash
# Motion Tracking 方式（使用提取的姿态）
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
  robot.asset.self_collisions=0 \
  headless=True
```

## 📈 预期训练结果

### 关键奖励指标

训练过程中应该看到以下指标逐渐提高：

```
rew_teleop_body_position_extend   目标: > 0.8  (身体位置跟踪)
rew_teleop_body_rotation          目标: > 0.9  (身体旋转跟踪)
rew_teleop_joint_position         目标: > 0.7  (关节角度跟踪)
rew_teleop_joint_velocity         目标: 接近0 (速度保持为0)
```

### 训练时长估计

- **快速测试**: ~1分钟
- **小规模验证**: ~30分钟
- **完整训练**: ~4-8小时（取决于GPU）

## 🔍 验证和测试

### 1. 可视化提取的姿态

```bash
python extract_standing_pose.py \
  --motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl \
  --time 0.0 \
  --output /tmp/test.pkl \
  --visualize
```

### 2. 测试训练好的模型

```bash
# 找到checkpoint文件
ls logs/StandingFromCR7/*/model_*.pt

# 运行评估
python humanoidverse/eval_agent.py \
  +checkpoint=logs/StandingFromCR7/<timestamp>-<exp_name>/model_2000.pt
```

## 📝 如何提取不同时刻的姿态

如果想从其他时刻提取站立姿态：

```bash
# 例如从 3.0 秒提取
python extract_standing_pose.py \
  --motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
  --time 3.0 \
  --output humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_3.0s.pkl \
  --visualize

# 然后更新训练命令中的 motion_file 路径
```

### 如何选择合适的提取时刻？

1. **查看原始动作视频**（如果有）
2. **尝试不同时刻**，观察提取的 `root_trans_offset[2]`（高度）：
   - 站立高度通常在 0.7-0.9m 之间
   - 太低（< 0.6m）可能是蹲下或坐姿
   - 太高（> 1.0m）可能是跳跃

3. **可视化验证**：
```bash
# 快速查看多个时刻
for t in 2.0 2.5 3.0 3.5; do
  python extract_standing_pose.py \
    --motion_file <motion_file> \
    --time $t \
    --output /tmp/pose_${t}s.pkl \
    --visualize
done
```

## 🆚 对比：Motion Tracking vs Standing 任务

| 特性 | Motion Tracking + 提取的姿态 | 自定义 Standing 任务 |
|------|---------------------------|---------------------|
| **使用提取的姿态** | ✅ 是 | ❌ 否 |
| **姿态来源** | 真实 CR7 动作 @ 2.8s | 奖励函数定义 |
| **精确度** | 🎯 完全复现 CR7 姿态 | 📐 学习通用站立 |
| **关节角度** | 固定（匹配提取值） | 灵活（接近默认值） |
| **训练难度** | 中等 | 较容易 |
| **适用场景** | 需要特定姿态 | 通用站立能力 |

**导师的要求**: 使用 Motion Tracking + 提取的姿态（方式1）✅

## ⚠️ 常见问题和解决方案

### Q1: 机器人直接倒下，无法站立

**检查清单**:
1. ✓ 提取的姿态高度合理？（~0.85m）
2. ✓ 关节角度在合理范围内？
3. ✓ 物理参数正确？

**解决方案**:
```bash
# 尝试不同时刻的姿态
python extract_standing_pose.py --time 2.5 ...
python extract_standing_pose.py --time 3.0 ...

# 或者放松终止条件
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.5

# 或者启用自碰撞
robot.asset.self_collisions=1
```

### Q2: 训练收敛慢

**可能原因**:
- 课程学习太激进
- 奖励权重不平衡

**解决方案**:
```bash
# 降低课程学习速度
rewards.reward_penalty_degree=0.000005
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.00001

# 增加训练步数
algo.num_steps_per_env=32  # 默认24
```

### Q3: 想要修改站立高度

**方式A**: 提取不同时刻的姿态（推荐）
```bash
# 尝试不同时刻，找到合适高度的姿态
python extract_standing_pose.py --time 2.5 ... --visualize
```

**方式B**: 手动修改提取的姿态文件
```python
import joblib
import numpy as np

data = joblib.load('standing_pose_from_CR7.pkl')
# 修改高度（z坐标）
data['motion0']['root_trans_offset'][0, 2] = 0.80  # 改为0.80米
joblib.dump(data, 'standing_pose_modified.pkl')
```

## 📊 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/StandingFromCR7

# 浏览器访问
http://localhost:6006
```

### 关键指标解释

```
Rewards/
  rew_teleop_body_position_extend    # 身体位置匹配度（越高越好）
  rew_teleop_body_rotation           # 身体旋转匹配度（越高越好）
  rew_teleop_joint_position          # 关节角度匹配度（越高越好）
  rew_penalty_action_rate            # 动作平滑度（接近0）
  rew_termination                    # 终止次数（越少越好）

Metrics/
  episode_length                     # 每episode持续时间（越长越好）
  total_reward                       # 总奖励（越高越好）
```

## 🎓 完整工作流程

```
1. 提取姿态
   ↓
   python extract_standing_pose.py --time 2.8 ...

2. 验证姿态
   ↓
   python extract_standing_pose.py ... --visualize

3. 快速测试
   ↓
   ./train_standing_from_cr7.sh  (选择模式1)

4. 小规模训练
   ↓
   ./train_standing_from_cr7.sh  (选择模式2)

5. 完整训练
   ↓
   ./train_standing_from_cr7.sh  (选择模式3)

6. 评估模型
   ↓
   python eval_agent.py +checkpoint=...

7. 部署使用
   ↓
   使用训练好的checkpoint进行推理
```

## 📚 相关文档索引

1. **`EXTRACT_STANDING_POSE_GUIDE.md`** - 详细的姿态提取指南
2. **`TRAIN_WITH_STANDING_POSE.md`** - 训练方法和参数详解
3. **`STANDING_TASK_README.md`** - Standing 任务文档
4. **`STANDING_VS_MOTION_TRACKING.md`** - 两种方法对比
5. **`QUICK_START_STANDING.md`** - 快速入门指南

## 🎉 总结

我们已经完成：
1. ✅ 从 CR7 动作提取站立姿态（2.8秒）
2. ✅ 创建完整的训练脚本和配置
3. ✅ 提供多种训练模式和参数
4. ✅ 编写详细的文档和指南

**现在可以开始训练了！** 推荐先运行快速测试模式验证一切正常：

```bash
./train_standing_from_cr7.sh
# 选择选项 1（快速测试）
```

如果快速测试成功看到机器人尝试保持站立姿态，就可以进行完整训练了！

---

**祝训练顺利！** 🚀

有任何问题请查阅相关文档或检查日志输出。

