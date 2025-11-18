# 📊 从Motion Tracking提取站立姿态 - 完整指南

## 🎯 Mentor的需求理解

### ❌ 你之前做的（独立训练站立）
- 从头训练一个站立策略
- 没有参考动作，机器人自己学习站立
- 类似Locomotion任务

### ✅ Mentor想要的（从参考动作提取）
- 从CR7动作文件中提取**特定时刻（2.8秒）**的姿态
- 这个姿态就是"站立"的标准姿态
- 用Motion Tracking任务跟踪这个**单帧姿态**
- 机器人学习保持这个特定的站立姿态

---

## 📋 两种数据的区别

### 1️⃣ Simulator状态（你代码里看到的）
```python
# 在 _post_physics_step() 中
root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
dof = self.simulator.dof_pos.cpu()
```
**这是什么**：仿真运行时，机器人的**实时状态**
**用途**：保存训练过程，用于分析机器人表现
**不是**：参考动作数据

### 2️⃣ Motion Library数据（Mentor想要的）
```python
# 从Motion Library加载参考动作
motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset)
root_pos = motion_res['root_pos']      # 参考动作的位置
root_rot = motion_res['root_rot']      # 参考动作的旋转
dof_pos = motion_res['dof_pos']        # 参考动作的关节角度
```
**这是什么**：CR7动作文件中**某一时刻的姿态数据**
**用途**：作为Motion Tracking的目标
**就是**：参考动作数据

---

## 🔧 实现步骤

### 步骤1：提取站立姿态

我已经为你创建了提取脚本：`extract_standing_pose.py`

```bash
python extract_standing_pose.py \
--motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
--time 2.8 \
--output humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl \
--visualize
```

**参数说明**：
- `--motion_file`: CR7动作文件路径
- `--time 2.8`: 提取2.8秒时刻的姿态（据mentor说这时是站立）
- `--output`: 输出文件名
- `--visualize`: 可选，查看提取的数据

**输出**：一个新的motion文件，只包含**一帧**数据（站立姿态）

---

### 步骤2：使用提取的站立姿态训练

**方法A：使用Motion Tracking任务**（推荐）

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
experiment_name=Standing_Extracted_From_Motion \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl"
```

**关键点**：
- ✅ 使用**Motion Tracking任务**（不是Standing任务）
- ✅ motion_file指向提取的站立姿态文件
- ✅ 机器人会学习**保持这个姿态**（因为只有一帧）

---

## 📊 工作原理图解

```
┌─────────────────────────────────────────────────────────────┐
│  CR7动作文件 (完整动作序列)                                  │
│  ├─ 0.0秒: 准备姿态                                          │
│  ├─ 1.0秒: 动作中                                            │
│  ├─ 2.0秒: 动作中                                            │
│  ├─ 2.8秒: ⭐ 站立姿态 ← 提取这一帧                         │
│  ├─ 3.0秒: 其他动作                                          │
│  └─ ...                                                      │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ extract_standing_pose.py
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  站立姿态文件 (单帧数据)                                      │
│  ├─ root_trans_offset: [x, y, z]                            │
│  ├─ root_rot: [qx, qy, qz, qw]                              │
│  ├─ dof: [23个关节角度]                                      │
│  ├─ root_lin_vel: [0, 0, 0]  ← 站立时速度为0                │
│  └─ dof_vel: [0, 0, ..., 0]  ← 站立时关节速度为0            │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Motion Tracking训练
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  机器人学习保持这个站立姿态                                   │
│  每一步都从Motion Library读取这个**相同的姿态**作为目标     │
│  ├─ Step 1: 目标 = 站立姿态                                  │
│  ├─ Step 2: 目标 = 站立姿态 (相同)                           │
│  ├─ Step 3: 目标 = 站立姿态 (相同)                           │
│  └─ ...                                                      │
│  结果：机器人学会保持这个特定的站立姿态                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🆚 对比：两种方法的区别

| 维度 | 独立Standing任务 | 从Motion提取站立 |
|------|----------------|-----------------|
| **参考目标** | 硬编码目标（height=0.78m等）| CR7动作的特定姿态 |
| **姿态来源** | 无特定姿态，机器人自由站立 | CR7在2.8秒的精确姿态 |
| **奖励函数** | 自定义站立奖励 | Motion Tracking奖励 |
| **训练任务** | Standing | Motion Tracking |
| **motion_file** | 不需要 | 需要（提取的单帧）|
| **优势** | 简单，容易调试 | **符合真实数据**，姿态标准 |
| **劣势** | 姿态不一定标准 | 需要提取步骤 |

---

## 🎯 为什么Mentor推荐这种方法？

1. **真实性**：CR7动作来自真实人类动作捕捉，姿态更自然
2. **一致性**：站立姿态与其他动作的姿态一致（都来自同一套数据）
3. **可扩展**：可以轻松提取其他姿态（如蹲下@3.5秒，抬手@4.2秒）
4. **标准化**：所有研究者使用相同的参考姿态，结果可比较

---

## 📝 实际操作流程

### 1. 检查原始动作文件

```python
import pickle

# 加载CR7动作文件
with open('humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看数据结构
print("Keys:", data.keys())
if 'motion0' in data:
    motion = data['motion0']
    print("FPS:", motion.get('fps', 50))
    print("Total frames:", motion['dof'].shape[0])
    print("Duration:", motion['dof'].shape[0] / motion.get('fps', 50), "seconds")
```

### 2. 运行提取脚本

```bash
python extract_standing_pose.py \
--motion_file <你的CR7文件路径> \
--time 2.8 \
--output humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose.pkl \
--visualize
```

### 3. 训练Motion Tracking任务

```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
... (其他参数) \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose.pkl"
```

### 4. 机器人会学习保持这个姿态

因为motion文件只有一帧，Motion Library会一直返回相同的姿态，机器人学习的就是**保持不动，维持这个姿态**。

---

## 🔍 核心代码理解

### Motion Tracking如何使用Motion Library

```python
# 在 _pre_compute_observations_callback() 中
motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

# motion_res 包含：
# - root_pos: 根位置
# - root_rot: 根旋转
# - dof_pos: 关节角度
# - body_vel_t: 身体速度
# - dof_vel: 关节速度

# 计算差异（用于奖励）
self.dif_global_body_pos = motion_res["rg_pos_t"] - self._rigid_body_pos_extend
self.dif_joint_angles = motion_res["dof_pos"] - self.simulator.dof_pos

# 奖励计算
reward = exp(-distance² / sigma)
```

**关键点**：
- 如果motion文件只有1帧，`motion_times`无论多少，都返回同一帧
- 机器人学习的目标就是：让当前姿态与这一帧尽可能接近
- 这就是"保持站立"

---

## ✅ 总结

**Mentor想要的方法**：
1. 从CR7动作文件中提取2.8秒时刻的姿态（这时CR7是站立的）
2. 保存为新的motion文件（只包含这一帧）
3. 用Motion Tracking任务训练，motion_file指向这个单帧文件
4. 机器人学习保持这个特定的站立姿态

**你之前做的方法**：
- 创建独立的Standing任务
- 定义站立目标（高度、双脚距离等）
- 从头训练站立策略

**两种方法都可以训练站立，但Mentor的方法更符合实际研究需求！**

---

## 🚀 立即开始

```bash
# 1. 提取站立姿态
python extract_standing_pose.py \
--motion_file humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
--time 2.8 \
--output humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose.pkl

# 2. 训练（使用Motion Tracking）
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
project_name=StandingFromMotion \
experiment_name=Standing_CR7_2.8s \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose.pkl"
```

完成！🎉

