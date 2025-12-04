# 更换参考动作数据集指南

本指南说明如何将项目中的 CR7 参考动作数据集更换为其他动作数据。

## ⚡ 快速参考

### 必需字段（每个动作字典必须包含）

```python
{
    'motion0': {
        'root_trans_offset': np.ndarray,  # [T, 3] 必需 - 根部（骨盆）位置，不是足部
        'pose_aa': np.ndarray,            # [T, num_joints, 3] 必需 - 姿态角轴表示（轴角向量）
        'fps': float,                     # 必需 - 帧率（推荐30fps）
    }
}
```

**重要**：只需要这3个字段！其他字段（如 `dof`, `root_rot` 等）会在加载时自动计算。

### `pose_aa` 核心概念（轴角表示）

**轴角（Axis-Angle）** 是一种3D旋转表示方法：
- **格式**：3维向量 `[rx, ry, rz]`
- **含义**：
  - 向量**方向**（归一化后）= 旋转轴
  - 向量**长度**（模）= 旋转角度（弧度）
- **例子**：
  - `[0, 0, 1.57]` = 绕Z轴旋转90度（π/2）
  - `[1.0, 0, 0]` = 绕X轴旋转57.3度（1弧度）
  - `[0, 0, 0]` = 无旋转
- **索引0** = 根部（骨盆）全局旋转
- **索引1+** = 其他关节相对父关节的局部旋转

**快速转换**（从四元数或欧拉角）：
```python
from scipy.spatial.transform import Rotation as sRot
# 从四元数: quat [x,y,z,w] -> 轴角
axis_angle = sRot.from_quat([0, 0, 0.707, 0.707]).as_rotvec()
# 从欧拉角: euler [x,y,z] -> 轴角
axis_angle = sRot.from_euler('xyz', [0, 0, np.pi/2]).as_rotvec()
```

### CR7数据参考

- **帧数 (T)**: 119帧
- **帧率 (fps)**: 30 fps
- **时长**: 3.97秒
- **根部高度**: z坐标约0.9米（骨盆位置，不是足部）

## 📋 目录

1. [概述](#概述)
2. [数据格式要求](#数据格式要求)
3. [更换步骤](#更换步骤)
4. [配置文件修改](#配置文件修改)
5. [训练脚本修改](#训练脚本修改)
6. [验证与测试](#验证与测试)

---

## 概述

项目中 CR7 数据集主要在以下场景使用：

1. **Motion Tracking 训练**：通过 `robot.motion.motion_file` 参数指定参考动作文件
2. **Sim2Real 部署**：在 `sim2real/config/g1_29dof_hist.yaml` 中配置动作类型和模型
3. **站立姿态提取**：从完整动作序列中提取特定时刻的姿态

### CR7 数据文件位置

- **完整动作序列**：
  ```
  humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl
  ```
- **提取的站立姿态**：
  ```
  humanoidverse/data/motions/g1_29dof_anneal_23dof/standing_pose_from_CR7.pkl
  ```

---

## 数据格式要求

### 1. 动作文件格式（.pkl）

动作数据文件必须是包含以下结构的字典。**重要**：每个动作字典只需要包含**必需字段**，其他字段会在加载时自动计算。

#### 必需字段（必须包含）

```python
{
    'motion0': {
        # ✅ 必需字段 1：根部位置偏移
        'root_trans_offset': np.ndarray,  # [T, 3] 形状，T为帧数，3为(x,y,z)坐标
        
        # ✅ 必需字段 2：姿态角轴表示
        'pose_aa': np.ndarray,            # [T, num_joints, 3] 形状
                                          # num_joints 为关节数量（包括根部）
                                          # 每个关节用3维角轴（axis-angle）表示旋转
        
        # ✅ 必需字段 3：帧率
        'fps': float,                     # 帧率（通常为 30 或 60）
    },
    'motion1': { ... },
    # ... 可以有多个动作序列
}
```

#### 可选字段（如果存在会被使用）

```python
{
    'motion0': {
        # ... 必需字段 ...
        
        # 可选字段 1：动作数据（如果存在，会在训练中使用）
        'action': np.ndarray,             # [T, action_dim] 动作序列
        
        # 可选字段 2：SMPL模型参数（如果使用SMPL人体模型）
        'beta': np.ndarray,               # [10] 或 [17] SMPL形状参数
        
        # 注意：以下字段不需要在原始数据中提供，
        # 它们会在 forward kinematics 过程中自动计算：
        # - 'dof': 关节角度（从 pose_aa 计算）
        # - 'dof_vel': 关节速度（从 dof 计算）
        # - 'root_rot': 根部旋转（从 pose_aa[0] 提取）
        # - 'root_lin_vel': 根部线速度（从 root_trans_offset 计算）
        # - 'root_ang_vel': 根部角速度（从 root_rot 计算）
    }
}
```

#### 自动计算字段是如何得出的？

MotionLib 拿到 `root_trans_offset`、`pose_aa` 和 `fps` 后，会通过机器人骨架的前向运动学（Forward Kinematics, FK）自动推导出其他量，无需你在数据里重复写入：

| 自动得到的字段 | 含义 | 计算方式 |
| --- | --- | --- |
| `root_rot` | 根部旋转 | 直接取 `pose_aa` 中根关节的轴角，转换为四元数 |
| `dof` / `dof_pos` | 机器人关节角度 | 根据骨架映射，把 `pose_aa` 投影到机器人各 DOF |
| `dof_vel` | 关节角速度 | 相邻帧 `dof` 之差 ÷ `dt`（`dt = 1 / fps`） |
| `root_lin_vel` | 根部线速度 | 相邻帧 `root_trans_offset` 之差 ÷ `dt` |
| `root_ang_vel` | 根部角速度 | 相邻帧 `root_rot` 的角度差 ÷ `dt` |
| `body_vel` / `body_ang_vel` / `rg_pos` 等 | 全身 marker / 链接的位姿与速度 | 基于骨架层级逐级做 FK 推导 |

因此，只要保持姿态与根轨迹正确，系统会自动补齐训练所需的全部运动学量。

#### `action` 字段是什么？如何在采集端记录？

- **是什么**：用于存储“控制器在每一帧输出了什么命令”，例如关节目标角、力矩、末端力或其他策略输出。
- **什么时候需要**：只有当你想在后续训练中复用这些命令（如残差学习、模仿已有控制器）时才需要写入，纯动捕驱动可以不写。
- **怎么记录**：
  1. 在采集脚本里，与动捕帧同步记录控制器当前帧的命令，组成 `[T, action_dim]` 的矩阵。
  2. 将该矩阵存入 `action` 字段，与 `root_trans_offset` / `pose_aa` 一起 `joblib.dump`。
  3. 若还有执行者 ID、接触标签、扰动标记等元数据，也可以自定义字段一并保存。

这样后续训练或分析就能直接读取原始控制信号，无需重新采集。

#### 字段说明

- **`root_trans_offset`** `[T, 3]`：
  - **根部（root）位置**：这是机器人的**根关节位置**，**不是足部位置**
  - 对于人形机器人，根部通常是**骨盆（pelvis）**或**腰部（waist）**位置
  - 在世界坐标系中的位置偏移，3 为 (x, y, z) 坐标
  - T 为时间帧数
  - 单位：米（m）
  - **注意**：z坐标通常是机器人的根部高度（站立时约0.7-1.0米），而不是地面高度（0米）

- **`pose_aa`** `[T, num_joints, 3]`：
  - **角轴（Axis-Angle）表示**：所有关节的旋转，使用**轴角**格式表示
  - **什么是轴角**：一个3维向量 `[rx, ry, rz]`，其中：
    - **方向**（向量归一化后）= 旋转轴
    - **长度**（向量的模）= 旋转角度（弧度）
    - 例如：`[0, 0, 1.57]` 表示绕Z轴旋转90度（π/2弧度）
  - **索引0是根部关节**：`pose_aa[:, 0, :]` 是根部（骨盆/腰部）的全局旋转
  - **索引1+是其他关节**：`pose_aa[:, 1:, :]` 是各个关节相对于父关节的局部旋转
  - `num_joints` 包括根部关节（索引0）和所有其他关节
  - 对于 `g1_29dof_anneal_23dof`，通常 `num_joints = 24`（1个根部 + 23个关节）
  - **与四元数的关系**：轴角可以通过 `scipy.spatial.transform.Rotation` 转换为四元数或旋转矩阵
  - **与关节角（DOF）的区别**：
    - `pose_aa` 是3D旋转的完整表示（每个关节3个自由度）
    - `dof` 是机器人实际可控的关节角度（通常1个自由度/关节，如肘关节只有屈伸）
    - 代码会从 `pose_aa` 投影计算出 `dof`

- **`fps`** `float`：
  - 帧率（frames per second）
  - 用于计算动作时长和插值
  - 常见值：30, 60
  - **CR7数据示例**：fps = 30，T = 119帧，时长 = 119/30 = 3.97秒

#### 最小示例

一个最简单的有效数据文件只需要包含这三个字段：

```python
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot

# 假设有 100 帧，24 个关节（1个根部 + 23个关节）
T = 100
num_joints = 24

motion_data = {
    'motion0': {
        'root_trans_offset': np.random.randn(T, 3).astype(np.float32),  # [100, 3]
        'pose_aa': np.random.randn(T, num_joints, 3).astype(np.float32),  # [100, 24, 3]
        'fps': 30.0,  # 30帧/秒
    }
}

# 保存
joblib.dump(motion_data, 'your_motion.pkl')
```

#### `pose_aa` 实际例子

如果你有四元数或欧拉角数据，需要转换为轴角格式：

```python
from scipy.spatial.transform import Rotation as sRot
import numpy as np

# 方法1: 从四元数转换
# 假设有一个四元数 [x, y, z, w] 表示旋转
quat = [0.0, 0.0, 0.707, 0.707]  # 绕Z轴旋转90度
rot = sRot.from_quat(quat)  # scipy使用 [x,y,z,w] 格式
axis_angle = rot.as_rotvec()  # 转换为轴角 [rx, ry, rz]
print(f"轴角: {axis_angle}")  # 输出: [0, 0, 1.57...] (约π/2)

# 方法2: 从欧拉角转换
euler_xyz = [0.0, 0.0, np.pi/2]  # 绕Z轴旋转90度 (XYZ顺序)
rot = sRot.from_euler('xyz', euler_xyz)
axis_angle = rot.as_rotvec()
print(f"轴角: {axis_angle}")  # 输出: [0, 0, 1.57...]

# 方法3: 直接构造轴角
# 轴角 = 旋转轴(归一化) × 旋转角度(弧度)
rotation_axis = np.array([0, 0, 1])  # Z轴
rotation_angle = np.pi / 2  # 90度
axis_angle = rotation_axis * rotation_angle
print(f"轴角: {axis_angle}")  # 输出: [0, 0, 1.57...]

# 用于 pose_aa 的完整示例
T = 100
num_joints = 24
pose_aa = np.zeros((T, num_joints, 3), dtype=np.float32)

# 设置根部旋转（第0个关节）- 例如机器人面向前方
for t in range(T):
    # 随时间轻微旋转根部
    angle = 0.1 * np.sin(2 * np.pi * t / T)  # 小幅摆动
    pose_aa[t, 0, :] = [0, 0, angle]  # 绕Z轴旋转

# 设置肘关节弯曲（假设是第5个关节）
for t in range(T):
    bend_angle = np.pi / 4 * (1 + np.sin(2 * np.pi * t / T))  # 45-90度之间
    # 假设肘关节绕Y轴弯曲
    pose_aa[t, 5, :] = [0, bend_angle, 0]

print(f"pose_aa 形状: {pose_aa.shape}")  # (100, 24, 3)
```

#### 理解轴角的直觉

轴角向量 `[rx, ry, rz]` 可以这样理解：

- **向量方向** = 旋转轴（穿过关节的虚拟杆）
- **向量长度** = 旋转多少弧度（右手定则）
- **零向量** `[0, 0, 0]` = 无旋转（初始姿态）

常见例子：
- `[1.57, 0, 0]`: 绕X轴旋转90度（俯仰）
- `[0, 1.57, 0]`: 绕Y轴旋转90度（偏航）
- `[0, 0, 1.57]`: 绕Z轴旋转90度（横滚）
- `[0, 0.785, 0]`: 绕Y轴旋转45度

**为什么使用轴角而不是欧拉角？**
- 轴角没有万向节锁（gimbal lock）问题
- 插值更平滑（可以直接线性插值小角度）
- 与SMPL等人体模型标准兼容
- 易于转换到其他格式（四元数、旋转矩阵）

### 2. 机器人配置要求

- **DOF 数量**：必须匹配机器人配置（`g1_29dof_anneal_23dof` 为 23 个关节）
- **坐标系**：根部位置和旋转需要符合项目约定
- **时间单位**：帧率（fps）用于计算动作时长

### 3. 数据预处理

如果您的数据是其他格式（如 BVH、FBX、AMASS 等），需要先转换为上述格式。可以参考项目中的数据处理脚本进行转换。

---

## 更换步骤

### 步骤 1：准备新的动作数据文件

1. **将新数据文件放置到合适位置**：
   ```bash
   # 建议放在以下目录之一：
   humanoidverse/data/motions/g1_29dof_anneal_23dof/          # 单文件
   humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/  # 多个文件
   ```

2. **确保文件格式正确**：
   - 文件扩展名为 `.pkl`
   - 数据结构符合上述要求
   - 数据维度匹配机器人配置

### 步骤 2：修改训练脚本

#### 方法 A：直接修改训练命令

在训练命令中直接指定新的动作文件路径：

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/your_new_motion.pkl" \
  # ... 其他参数
```

#### 方法 B：修改 `train_standing_from_cr7.sh`

编辑 `train_standing_from_cr7.sh`，在相应的选项中添加新的动作文件路径：

```bash
# 例如，添加选项 6
6)
    MOTION_TYPE="YourNewMotion"
    MOTION_FILE="humanoidverse/data/motions/g1_29dof_anneal_23dof/your_new_motion.pkl"
    MOTION_DESC="您的新动作描述"
    RESAMPLE_MOTION=True
    PROJECT_NAME="YourProjectName"
    MOTION_SUFFIX="YourMotion"
    REWARD_CONFIG="motion_tracking/reward_motion_tracking_walking"
    ;;
```

### 步骤 3：更新配置文件（可选）

如果需要在 Sim2Real 部署中使用新动作，需要修改 `sim2real/config/g1_29dof_hist.yaml`：

#### 3.1 添加动作类型映射

在 `mimic_robot_types` 中添加新条目：

```yaml
mimic_robot_types: {
  "CR7_level1": "g1_29dof_anneal_23dof",
  "YourNewMotion_level1": "g1_29dof_anneal_23dof",  # 新增
  # ...
}
```

#### 3.2 添加模型映射（如果已有训练好的模型）

在 `mimic_models` 中添加：

```yaml
mimic_models: {
  "CR7_level1": "model_191500.onnx",
  "YourNewMotion_level1": "model_XXXXX.onnx",  # 新增
  # ...
}
```

#### 3.3 添加初始姿态（可选）

在 `start_upper_body_dof_pos` 中添加新动作的初始上半身姿态：

```yaml
start_upper_body_dof_pos: {
  "CR7_level1": [0.2049, -0.1423, ...],  # 17个值
  "YourNewMotion_level1": [0.0, 0.0, ...],  # 新增，17个值
  # ...
}
```

#### 3.4 添加动作时长

在 `motion_length_s` 中添加：

```yaml
motion_length_s: {
  "CR7_level1": 3.967,
  "YourNewMotion_level1": 5.0,  # 新增，单位为秒
  # ...
}
```

---

## 配置文件修改

### 完整示例：添加新动作 "Dance"

假设您有一个名为 "Dance" 的新动作，需要完整配置：

#### 1. 修改 `g1_29dof_hist.yaml`

```yaml
# 在 mimic_robot_types 中添加
mimic_robot_types: {
  "Dance_level1": "g1_29dof_anneal_23dof",
}

# 在 mimic_models 中添加（训练后）
mimic_models: {
  "Dance_level1": "model_200000.onnx",
}

# 在 start_upper_body_dof_pos 中添加初始姿态
start_upper_body_dof_pos: {
  "Dance_level1": [
    0.0, 0.0, 0.0,  # waist (3个值)
    0.0, 0.3, 0.0, 1.0,  # left shoulder and elbow (4个值)
    0.0, 0.0, 0.0,  # left wrist (3个值)
    0.0, -0.3, 0.0, 1.0,  # right shoulder and elbow (4个值)
    0.0, 0.0, 0.0  # right wrist (3个值)
  ],  # 总共17个值
}

# 在 motion_length_s 中添加时长
motion_length_s: {
  "Dance_level1": 8.5,  # 秒
}
```

#### 2. 准备动作文件

将 `dance_motion.pkl` 放置在：
```
humanoidverse/data/motions/g1_29dof_anneal_23dof/dance_motion.pkl
```

#### 3. 训练命令

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/dance_motion.pkl" \
  project_name=DanceMotionTracking \
  experiment_name=Dance_L1 \
  # ... 其他参数
```

---

## 训练脚本修改

### 修改 `train_standing_from_cr7.sh` 添加新选项

在脚本的选项部分添加新动作：

```bash
# 在 case 语句中添加
6)
    MOTION_TYPE="Dance"
    MOTION_FILE="humanoidverse/data/motions/g1_29dof_anneal_23dof/dance_motion.pkl"
    MOTION_DESC="舞蹈动作"
    RESAMPLE_MOTION=True
    PROJECT_NAME="DanceMotionTracking"
    MOTION_SUFFIX="Dance"
    REWARD_CONFIG="motion_tracking/reward_motion_tracking_walking"
    ;;
```

同时更新提示信息：

```bash
echo "  6) 舞蹈动作 (Dance)"
```

---

## 验证与测试

### 1. 检查数据文件

使用 Python 脚本验证数据格式（**只检查必需字段**）：

```python
import joblib
import numpy as np

# 加载数据
data = joblib.load('your_motion.pkl')

# 检查必需字段
required_fields = ['root_trans_offset', 'pose_aa', 'fps']

print("动作数据文件验证")
print("=" * 50)
print(f"包含的动作数量: {len(data)}")
print(f"动作键名: {list(data.keys())}")

for key in data.keys():
    motion = data[key]
    print(f"\n{key}:")
    
    # 检查必需字段
    for field in required_fields:
        if field in motion:
            value = motion[field]
            if isinstance(value, np.ndarray):
                print(f"  ✅ {field}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  ✅ {field}: {value}")
        else:
            print(f"  ❌ 缺少必需字段: {field}")
    
    # 检查可选字段
    optional_fields = ['action', 'beta']
    for field in optional_fields:
        if field in motion:
            value = motion[field]
            if isinstance(value, np.ndarray):
                print(f"  📦 {field} (可选): {value.shape}")
            else:
                print(f"  📦 {field} (可选): {value}")
    
    # 验证数据维度
    if 'pose_aa' in motion and 'root_trans_offset' in motion:
        T_pose = motion['pose_aa'].shape[0]
        T_trans = motion['root_trans_offset'].shape[0]
        if T_pose != T_trans:
            print(f"  ⚠️  警告: 帧数不匹配 (pose_aa: {T_pose}, root_trans_offset: {T_trans})")
        else:
            print(f"  ✅ 帧数一致: {T_pose} 帧")
        
        num_joints = motion['pose_aa'].shape[1]
        print(f"  ✅ 关节数量: {num_joints} (应该为 24: 1个根部 + 23个关节)")

print("\n" + "=" * 50)
print("验证完成！")
```

### 2. 快速测试训练

使用小规模配置快速验证：

```bash
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  robot.motion.motion_file="your_motion.pkl" \
  num_envs=1 \
  headless=False \
  algo.config.num_learning_iterations=10
```

### 3. 检查日志

训练开始后，检查：
- 动作是否正确加载
- 观察值维度是否正确
- 奖励函数是否正常计算

---

## 常见问题

### Q1: 每个动作字典需要包含所有字段吗？

**答案**：**不需要**。每个动作字典只需要包含**3个必需字段**：

1. ✅ `root_trans_offset` - 根部位置
2. ✅ `pose_aa` - 姿态角轴表示
3. ✅ `fps` - 帧率

其他字段（如 `dof`, `root_rot`, `dof_vel` 等）会在数据加载时通过 forward kinematics 自动计算，**不需要在原始数据文件中提供**。

**为什么？**
- 代码会从 `pose_aa` 和 `root_trans_offset` 计算出所有需要的关节位置、旋转、速度等信息
- 这样可以减少数据文件大小，并确保数据一致性

**可选字段**：
- `action` - 如果存在，会在训练中使用
- `beta` - 如果使用 SMPL 模型

### Q2: 根部位置是足部位置吗？

**答案**：**不是**。根部位置是机器人的**根关节位置**，通常是：

- **骨盆（pelvis）**或**腰部（waist）**位置
- 这是整个机器人骨架的根节点
- 站立时，根部高度通常在 **0.7-1.0米** 左右（取决于机器人尺寸）
- **足部位置**是末端执行器，需要通过 forward kinematics 从根部位置计算得出

**CR7数据示例**：
- `root_trans_offset` 的 z 坐标约为 **0.9米**，这是根部（骨盆）的高度
- 足部位置会根据腿部姿态变化，通常在地面附近（接近0米）

### Q3: CR7数据的时间帧数和帧率是多少？

**CR7实际数据**：
- **帧数 (T)**: 119 帧
- **帧率 (fps)**: 30 fps
- **实际时长**: 119 / 30 = **3.97秒**
- **pose_aa形状**: (119, 27, 3) - 27个关节（1个根部 + 26个其他关节）
- **root_trans_offset形状**: (119, 3)

这与配置文件 `g1_29dof_hist.yaml` 中的 `motion_length_s["CR7_level1"] = 3.967` 一致。

### Q4: 如何合适地设置T和fps？

**设置原则**：

1. **帧率 (fps) 选择**：
   - **30 fps**：适合大多数动作，平衡质量和文件大小（**推荐**）
   - **60 fps**：适合快速动作（如跳跃、踢球），更平滑但文件更大
   - **不要低于 20 fps**：否则动作会显得不连贯

2. **帧数 (T) 计算**：
   ```
   T = 动作时长（秒）× fps
   ```
   - 例如：4秒的动作，30fps → T = 4 × 30 = 120帧
   - 例如：3.97秒的动作，30fps → T = 3.97 × 30 ≈ 119帧（如CR7）

3. **实际建议**：
   - **短动作**（1-5秒）：30 fps，T = 30-150帧
   - **中等动作**（5-10秒）：30 fps，T = 150-300帧
   - **长动作**（10秒以上）：30 fps，T = 300+帧
   - **快速动作**（如踢球、跳跃）：建议60 fps以获得更平滑的轨迹

4. **验证方法**：
   ```python
   # 检查动作时长是否合理
   T = 119  # 帧数
   fps = 30  # 帧率
   时长 = T / fps  # 应该接近实际动作时长
   print(f"动作时长: {时长:.2f} 秒")
   ```

### Q5: 数据维度不匹配

**问题**：`pose_aa` 的关节数量与机器人配置不一致

**解决**：
- 检查 `pose_aa` 的第二个维度（`num_joints`）
- 对于 `g1_29dof_anneal_23dof`，`num_joints` 应该为 24（1个根部 + 23个关节）
- 注意：CR7数据使用27个关节，这是正常的，代码会自动处理映射
- 确保 `pose_aa` 的形状为 `[T, num_joints, 3]`

### Q6: 动作时长不正确

**问题**：动作播放速度异常

**解决**：
- 检查 `fps` 字段是否正确
- 确认 `motion_length_s` 配置与实际数据匹配
- 计算公式：`时长 = 帧数 / fps`

### Q7: 初始姿态不匹配

**问题**：机器人初始姿态与参考动作差异大

**解决**：
- 检查 `start_upper_body_dof_pos` 配置
- 确保初始姿态接近动作序列的第一帧
- 可以提取动作第一帧作为初始姿态

### Q8: 坐标系问题

**问题**：机器人朝向或位置错误

**解决**：
- 检查 `root_rot` 的格式（xyzw vs wxyz）
- 确认根部位置偏移的坐标系
- 参考现有 CR7 数据的坐标系约定

### Q9: `action` 字段需要怎么准备？

- **用途**：保存控制器/策略在每一帧输出的命令（关节角、力矩、末端力等），方便后续训练复现或做残差学习。
- **是否必需**：不是必需；只有当你希望在训练中使用这些命令时才需要写入。
- **采集方法**：在采集脚本里，与动捕帧同步记录控制命令，组成 `[T, action_dim]` 的数组并存入 `action` 字段；若没有控制信号，可直接忽略该字段。

### Q10: `pose_aa` 与机器人实际关节角（DOF）有什么关系？

**核心区别**：

| 概念 | `pose_aa` | `dof` (关节角) |
|------|-----------|----------------|
| **定义** | 3D空间中的完整旋转表示 | 机器人实际可控的关节角度 |
| **维度** | 每个关节3个值（轴角向量） | 每个关节1个值（通常） |
| **自由度** | 理论上3自由度（任意旋转） | 实际受机械限制（如肘关节只能屈伸） |
| **用途** | 描述人体/动捕数据的姿态 | 控制机器人电机 |
| **例子** | `[0.5, 0.3, 0.1]` 表示复杂的3D旋转 | `1.2` 弧度表示肘关节弯曲角度 |

**转换过程**：
```
动捕数据 pose_aa [T, 24, 3]
    ↓ (通过骨架映射和投影)
机器人关节角 dof [T, 23]
    ↓ (通过PD控制器)
电机命令
```

**实际代码中的转换**（在 `motion_lib_base.py` 和 `fit_smpl_motion.py` 中）：
1. **Forward Kinematics (FK)**：从 `pose_aa` + `root_trans_offset` 计算全身各关节的3D位置
2. **Inverse Kinematics (IK) 或优化**：找到最接近目标位置的机器人关节角 `dof`
3. **关节映射**：将人体骨架的关节映射到机器人关节（如 SMPL 的 "L_Elbow" → 机器人的 "left_elbow"）

**为什么需要这个转换？**
- 人体模型（如SMPL）使用3D旋转描述所有关节（适合动捕）
- 机器人有机械限制，很多关节只有1个自由度（如膝盖、肘部）
- 需要从理论姿态提取出机器人能实际执行的命令

**示例**：
```python
# 人体肘关节的 pose_aa（可以有复杂的3D旋转）
pose_aa_elbow = [0.3, 1.2, 0.1]  # 3个值

# 转换后机器人肘关节只有屈伸
dof_elbow = 1.2  # 1个值（弯曲角度）
```

---

## 快速参考

### 关键文件位置

| 文件 | 路径 | 作用 |
|------|------|------|
| 训练脚本 | `train_standing_from_cr7.sh` | 训练入口，指定动作文件 |
| 配置文件 | `sim2real/config/g1_29dof_hist.yaml` | Sim2Real 部署配置 |
| 动作数据目录 | `humanoidverse/data/motions/g1_29dof_anneal_23dof/` | 动作文件存储位置 |
| 训练入口 | `humanoidverse/train_agent.py` | 训练主程序 |

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `robot.motion.motion_file` | 动作文件路径 | `"humanoidverse/data/motions/.../motion.pkl"` |
| `env.config.resample_motion_when_training` | 是否重采样动作 | `True` / `False` |
| `project_name` | 项目名称 | `"MotionTracking"` |
| `experiment_name` | 实验名称 | `"YourExperiment"` |

---

## 总结

更换 CR7 数据集为其他数据的核心步骤：

1. ✅ **准备数据**：确保 `.pkl` 文件格式正确
2. ✅ **修改训练命令**：通过 `robot.motion.motion_file` 指定新文件
3. ✅ **更新配置**（可选）：在 `g1_29dof_hist.yaml` 中添加新动作类型
4. ✅ **验证测试**：使用小规模配置快速验证

如有问题，请参考项目中的其他动作数据文件作为示例，或查看相关文档。

