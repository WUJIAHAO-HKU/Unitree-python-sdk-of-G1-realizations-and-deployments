# Motion Tracking 训练脚本使用指南

## 🚀 快速开始

```bash
./train_standing_from_cr7.sh
```

## 📋 使用流程

### 步骤 1: 选择参考动作类型

```
1) 站立姿态 (Standing Pose)
   - 来源: 从CR7动作的2.8秒时刻提取
   - 用途: 训练机器人保持稳定的站立姿态
   - 重采样: 否 (单帧姿态)
   - 项目名: StandingFromCR7

2) 走路动作 Level 1 (简单)
   - 文件: walk_level1_filter_amass.pkl
   - 难度: ⭐
   - 重采样: 是 (动作序列)
   - 项目名: WalkingMotionTracking

3) 走路动作 Level 2 (中等)
   - 文件: walk_level2_filter_amass.pkl
   - 难度: ⭐⭐
   - 重采样: 是
   - 项目名: WalkingMotionTracking

4) 走路动作 Level 3 (较难)
   - 文件: walk_level3_filter_amass.pkl
   - 难度: ⭐⭐⭐
   - 重采样: 是
   - 项目名: WalkingMotionTracking

5) 走路动作 Level 4 (困难)
   - 文件: walk_level4_filter_amass.pkl
   - 难度: ⭐⭐⭐⭐
   - 重采样: 是
   - 项目名: WalkingMotionTracking
```

### 步骤 2: 选择训练模式

```
1) 快速测试
   - 环境数: 1
   - 可视化: 开启 (可观看训练过程)
   - 迭代次数: 10
   - 用途: 快速验证脚本是否正常工作

2) 小规模验证
   - 环境数: 256
   - 可视化: 关闭 (无头模式)
   - 迭代次数: 100
   - 用途: 验证参数设置和初步效果

3) 完整训练
   - 环境数: 4096
   - 可视化: 关闭
   - 迭代次数: 1000
   - 用途: 获得最佳训练效果
```

## 📊 训练建议

### 站立姿态训练
- **推荐迭代次数**: 100-500
- **收敛时间**: 较快
- **关键指标**: 
  - 姿态误差 < 0.1
  - 不倒下即为成功
- **日志位置**: `logs/StandingFromCR7/`

### 走路动作训练
- **Level 1 推荐迭代**: 500-1000
- **Level 2-4 推荐迭代**: 1000-2000+
- **收敛时间**: 较慢
- **关键指标**:
  - 动作跟踪误差逐渐降低
  - 能完整执行走路动作序列
- **日志位置**: `logs/WalkingMotionTracking/`

## 🎮 训练后测试

查看训练完成后的输出信息，使用提供的命令：

```bash
# 查看训练曲线
tensorboard --logdir logs/<PROJECT_NAME>

# 测试模型
python humanoidverse/eval_agent.py \
  +checkpoint=logs/<PROJECT_NAME>/<timestamp>-<EXP_NAME>/model_XXXX.pt
```

## ⚙️ 关键参数说明

### 动作重采样 (resample_motion_when_training)
- **False**: 用于单帧姿态（如站立）
- **True**: 用于动作序列（如走路）

### 终止条件
- `terminate_when_motion_far=True`: 距离参考动作太远时终止
- `terminate_by_low_height=True`: 机器人高度过低时终止
- `terminate_by_gravity=True`: 机器人倾斜过大时终止

## 🔍 示例使用场景

### 场景 1: 快速测试走路 Level 1
```
选择: 动作类型 = 2 (走路 Level 1)
选择: 训练模式 = 1 (快速测试)
结果: 可视化窗口中观看机器人学习走路
```

### 场景 2: 完整训练站立姿态
```
选择: 动作类型 = 1 (站立姿态)
选择: 训练模式 = 3 (完整训练)
结果: 4096环境并行训练，1000次迭代
```

### 场景 3: 挑战困难走路动作
```
选择: 动作类型 = 5 (走路 Level 4)
选择: 训练模式 = 3 (完整训练)
结果: 训练高难度走路，需要更多迭代
```

## 💡 提示与技巧

1. **首次使用**: 建议先用"快速测试"模式熟悉流程
2. **站立姿态**: 如果站立姿态文件不存在，脚本会提示如何提取
3. **走路难度**: 建议按 Level 1 → 2 → 3 → 4 顺序递进训练
4. **观察训练**: 快速测试模式可以直观看到机器人的学习过程
5. **日志管理**: 不同项目名会创建不同的日志文件夹，便于管理

## 🛠️ 故障排除

### 动作文件不存在
```
❌ 错误: 动作文件不存在
解决: 检查文件路径，或提取站立姿态文件
```

### 训练不收敛
```
原因: 
  - 动作难度过高
  - 迭代次数不足
  - 参数设置不当
解决:
  - 降低难度等级
  - 增加迭代次数
  - 调整奖励权重
```

## 📁 文件结构

```
ASAP/
├── train_standing_from_cr7.sh          # 主训练脚本
├── humanoidverse/
│   └── data/
│       └── motions/
│           └── g1_29dof_anneal_23dof/
│               ├── standing_pose_from_CR7.pkl     # 站立姿态
│               └── TairanTestbed/singles/
│                   ├── 0-..._walk_level1_....pkl  # 走路L1
│                   ├── 0-..._walk_level2_....pkl  # 走路L2
│                   ├── 0-..._walk_level3_....pkl  # 走路L3
│                   └── 0-..._walk_level4_....pkl  # 走路L4
└── logs/
    ├── StandingFromCR7/          # 站立训练日志
    └── WalkingMotionTracking/    # 走路训练日志
```

## 🎯 总结

这个脚本提供了灵活的训练选项：
- ✅ 5种动作类型（1站立 + 4走路难度）
- ✅ 3种训练规模（测试/验证/完整）
- ✅ 自动配置参数（重采样、项目名等）
- ✅ 友好的交互式界面
- ✅ 详细的训练信息输出

开始您的Motion Tracking训练之旅吧！🚀

