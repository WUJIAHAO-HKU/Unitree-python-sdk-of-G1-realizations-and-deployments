# 📚 Standing Task - 文档索引

## 🚀 快速开始

如果你是第一次使用，建议按以下顺序阅读：

1. **[快速开始指南](QUICK_START_STANDING.md)** ⚡
   - 一键训练命令
   - 快速调优参考
   - 成功标准

2. **[完整教程](STANDING_TASK_README.md)** 📖
   - 详细的使用说明（23个章节）
   - 所有参数的含义
   - 故障排查

3. **[对比分析](STANDING_VS_MOTION_TRACKING.md)** 🔍
   - Standing vs Motion Tracking详细对比
   - 15个维度的差异
   - 代码逻辑对比

4. **[实现总结](STANDING_TASK_SUMMARY.md)** 📝
   - 完整的数据提取清单
   - 架构设计说明
   - 独立性保证

5. **[架构图](ARCHITECTURE_DIAGRAM.md)** 🏗️
   - 可视化的文件结构
   - 数据流图
   - 运行时架构

---

## 📂 文档清单

### 主要文档（5个）

| 文档名 | 用途 | 页数估计 | 推荐阅读顺序 |
|--------|------|----------|-------------|
| `QUICK_START_STANDING.md` | 快速开始 | 1页 | 1️⃣ 第一个 |
| `STANDING_TASK_README.md` | 完整教程 | 10页 | 2️⃣ 深入了解 |
| `STANDING_VS_MOTION_TRACKING.md` | 对比分析 | 8页 | 3️⃣ 理解差异 |
| `STANDING_TASK_SUMMARY.md` | 实现总结 | 6页 | 4️⃣ 技术细节 |
| `ARCHITECTURE_DIAGRAM.md` | 架构图 | 5页 | 5️⃣ 系统架构 |

### 脚本工具（1个）

| 文件名 | 用途 | 执行方式 |
|--------|------|----------|
| `test_standing_task.sh` | 配置测试脚本 | `./test_standing_task.sh` |

---

## 🎯 按需查找

### 我想...

#### 🚀 立即开始训练
→ 阅读：[QUICK_START_STANDING.md](QUICK_START_STANDING.md)
→ 一键命令：
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

#### 📖 了解所有细节
→ 阅读：[STANDING_TASK_README.md](STANDING_TASK_README.md)
→ 涵盖：
- 任务概述
- 训练目标
- 奖励机制（12个奖励项）
- 观测空间（80维）
- 终止条件（4个）
- 参数调优
- 故障排查
- 进阶使用

#### 🔍 理解与Motion Tracking的区别
→ 阅读：[STANDING_VS_MOTION_TRACKING.md](STANDING_VS_MOTION_TRACKING.md)
→ 对比：
- 训练目标
- 奖励机制
- 观测空间
- 终止条件
- 课程学习
- 核心代码
- 文件映射
- 训练效率
- 应用场景

#### 📊 查看数据提取结果
→ 阅读：[STANDING_TASK_SUMMARY.md](STANDING_TASK_SUMMARY.md)
→ 包含：
- 已完成工作清单
- 数据提取清单
- 与Motion Tracking对比表
- 预期训练效果
- 独立性保证

#### 🏗️ 理解系统架构
→ 阅读：[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
→ 展示：
- 文件目录结构
- 类继承关系
- 数据流图
- 奖励计算流程
- 配置文件依赖
- 运行时架构

#### 🧪 测试配置是否正确
→ 执行：`./test_standing_task.sh`
→ 会进行：
- 配置加载测试
- 短时间训练测试（1个iteration）
- 输出测试结果

#### 🔧 调整奖励权重
→ 阅读：[STANDING_TASK_README.md](STANDING_TASK_README.md) 第9节
→ 编辑：`humanoidverse/config/rewards/standing/reward_standing_g1.yaml`
→ 参考：
```yaml
# 容易摔倒？增大这些
standing_upright: 3.0          # 原来 2.0
penalty_orientation: -3.0      # 原来 -2.0

# 站立不稳？增大这些
standing_zero_velocity: 1.5    # 原来 0.8
penalty_lin_vel: -2.0          # 原来 -1.5
```

#### 📊 监控训练进度
→ 阅读：[STANDING_TASK_README.md](STANDING_TASK_README.md) 第17节
→ 命令：`tensorboard --logdir=logs/Standing`
→ 访问：http://localhost:6006
→ 关注指标：
- `Train/mean_reward` (应该持续上升)
- `Train/mean_episode_length` (应该接近400)
- `Loss/Surrogate` (应该逐渐收敛)
- `Loss/Value` (应该逐渐减小)

#### 🐛 解决训练问题
→ 阅读：[STANDING_TASK_README.md](STANDING_TASK_README.md) 第18节
→ 常见问题：
- ImportError
- 奖励一直是负数
- 机器人一直摔倒
- 训练很慢

#### 🎮 评估训练好的模型
→ 阅读：[QUICK_START_STANDING.md](QUICK_START_STANDING.md)
→ 命令：
```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/Standing/<run_dir>/model_5000.pt
```

---

## 📁 代码文件位置

### 核心任务实现
```
humanoidverse/envs/standing/
├── __init__.py           # 模块初始化
└── standing.py           # LeggedRobotStanding类（250行）
```

### 配置文件
```
humanoidverse/config/
├── exp/standing.yaml                           # 实验配置
├── env/standing.yaml                           # 环境配置
├── obs/standing/standing_obs_basic.yaml        # 观测配置
└── rewards/standing/reward_standing_g1.yaml    # 奖励配置
```

### 文档文件
```
ASAP/
├── QUICK_START_STANDING.md           # 快速开始
├── STANDING_TASK_README.md           # 完整教程
├── STANDING_VS_MOTION_TRACKING.md    # 对比分析
├── STANDING_TASK_SUMMARY.md          # 实现总结
├── ARCHITECTURE_DIAGRAM.md           # 架构图
├── STANDING_TASK_INDEX.md            # 本索引文件
└── test_standing_task.sh             # 测试脚本
```

---

## 🎓 学习路径

### 入门级（只想快速训练）
1. 阅读 [QUICK_START_STANDING.md](QUICK_START_STANDING.md)
2. 执行训练命令
3. 监控 TensorBoard
4. 评估模型

### 中级（想理解原理）
1. 阅读 [STANDING_TASK_README.md](STANDING_TASK_README.md)
2. 理解奖励机制
3. 理解观测空间
4. 理解终止条件
5. 尝试调优参数

### 高级（想深入研究）
1. 阅读 [STANDING_VS_MOTION_TRACKING.md](STANDING_VS_MOTION_TRACKING.md)
2. 阅读 [STANDING_TASK_SUMMARY.md](STANDING_TASK_SUMMARY.md)
3. 阅读 [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
4. 研究源代码 `standing.py`
5. 修改和扩展任务

---

## 🔗 相关资源

### 代码库
- 主仓库：`/home/wujiahao/ASAP`
- 环境：`conda activate hvgym`

### 依赖项
- Isaac Gym
- PyTorch
- Hydra
- TensorBoard

### 训练日志
- 默认路径：`logs/Standing/`
- 包含：模型checkpoint、TensorBoard日志、配置文件

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] 阅读了 [QUICK_START_STANDING.md](QUICK_START_STANDING.md)
- [ ] conda环境已激活：`conda activate hvgym`
- [ ] Isaac Gym可用
- [ ] GPU可用（`nvidia-smi`检查）
- [ ] 磁盘空间充足（至少10GB）

---

## 📞 获取帮助

1. **配置错误**：检查 [STANDING_TASK_README.md](STANDING_TASK_README.md) 第18节
2. **训练问题**：检查 [STANDING_TASK_README.md](STANDING_TASK_README.md) 第18节
3. **理解差异**：阅读 [STANDING_VS_MOTION_TRACKING.md](STANDING_VS_MOTION_TRACKING.md)
4. **架构问题**：阅读 [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)

---

## 🎉 开始你的Standing训练之旅！

选择一个起点：
- 🚀 **快速开始** → [QUICK_START_STANDING.md](QUICK_START_STANDING.md)
- 📖 **深入学习** → [STANDING_TASK_README.md](STANDING_TASK_README.md)
- 🔍 **理解差异** → [STANDING_VS_MOTION_TRACKING.md](STANDING_VS_MOTION_TRACKING.md)

祝训练顺利！ 🤖✨

