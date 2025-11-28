# Sim2Real 残差学习手册

本分支 `sim2real` 只关注“动捕 → 数据采集 → 残差学习 → Unitree 实机部署”整体链路。若需要完整版 ASAP 说明，请参阅主仓库历史版本。

---

## 1. 项目定位
- **目标**：让 G1（`g1_29dof_anneal_23dof`）在真实环境中复现高自由度动捕动作，并通过残差策略对基准控制器补偿。
- **核心思路**：利用动捕服提供的人体轨迹，Unitree SDK 提供的低级控制接口，以及 RL/Residual policy，在仿真-现实之间形成闭环。
- **成果**：
  - `MOCAP_RESIDUAL_LEARNING_MINDMAP.md`：总体思维导图 + 数据要素。
  - `sim2real/rl_policy/listener_deltaa.py`：高频数据采集与日志工具。
  - `humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml`：仿真/控制参数真值参考。

---

## 2. 关键目录速览
| 路径 | 作用 |
| --- | --- |
| `humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml` | G1 机器人 DOF、关节限制、控制增益等真值定义，是仿真和部署的一致性基础 |
| `sim2real/rl_policy/listener_deltaa.py` | ROS2 + Unitree SDK 数据记录节点，整合低级状态、命令与动捕 `/odometry` |
| `humanoidverse/train_agent.py` | 训练入口，可加载 delta-action / motion-tracking 配置 |
| `sim2real/utils/unitree_sdk2py_bridge.py` | Unitree SDK V2 Python 接口封装 |
| `logs/`, `humanoidverse/logs/` | 训练与真实采集输出目录 |

---

## 3. Sim2Real 系统拓扑
1. **动捕侧**：
   - 采集骨架姿态、质心、末端速度、接触事件。
   - 话题 `/odometry` 或自定义 Topic，为 listener 提供参考轨迹。
2. **机器人侧**：
   - Unitree 低级接口 `rt/lowstate`（编码器、IMU、估计力矩）。
   - 控制命令 `rt/lowcmd`（目标位置/速度/KP/KD/力矩）。
3. **同步与缓存**：
   - `listener_deltaa.py` 以 200 Hz 拉齐三个通道，并在按键触发时写入 `.npz`。
4. **离线训练**：
   - 以 `g1_29dof_anneal_23dof.yaml` 为真值约束，构建残差标签（基准命令 vs 实际执行 vs 动捕参考）。
5. **在线部署**：
   - 残差策略在 Unitree 上以附加力矩/角度方式执行，安全限幅由 YAML 参数和控制器共同约束。

---

## 4. 动捕驱动的残差学习流程
详细思路可见 `MOCAP_RESIDUAL_LEARNING_MINDMAP.md`，这里提炼执行步骤：

1. **定义补偿目标**
   - 明确基准控制器类型（PD / delta action）。
   - 设定残差输出形式：Δτ、Δq、Δ末端力中的一种。

2. **设计动作覆盖**
   - 站立、步行、转弯、上下坡、扰动、载荷等场景。
   - 记录环境标签（摩擦、坡度、外力）。

3. **同步采集**
   - 动捕服骨架（position/orientation/vel/ang vel）。
   - Unitree `lowstate`（23 DOF 的 q/dq/tau + IMU）。
   - Unitree `lowcmd`（控制器输出 q/dq/kp/kd/tau）。
   - 时间戳统一并处理延迟。

4. **数据清洗与映射**
   - 滤波、异常点剔除、坐标系转换（人体 → 机器人 DOF）。
   - 标记接触相位、地形条件。

5. **构建残差标签**
   - `残差 = 动捕参考驱动量 - 基准控制落地效果`。
   - 结合 YAML 中的 DOF 限制，裁剪不可执行部分。

6. **模型训练 / 验证**
   - 仿真重放 → 验证稳定性。
   - 真实机部署时加入安全限幅、在线日志回采。

7. **快速检查**
   - 信号是否齐全？
   - 同步误差是否 <5 ms？
   - 数据集是否覆盖全部动作相位？

---

## 5. `listener_deltaa.py` 解析
文件：`sim2real/rl_policy/listener_deltaa.py`

- **Node 角色**：ROS2 `DataLogger`，整合动捕 `/odometry` 与 Unitree DDS 低级接口。
- **核心逻辑**：
  1. 初始化 `ChannelFactoryInitialize`（依据配置选择网卡/域）。
  2. 订阅 `rt/lowstate`、`rt/lowcmd`，解析前 `NUM_JOINTS`（根据 `Robot(config)` 自动适配 G1 23 DOF）。
  3. 订阅 `Odometry`，提取位姿与 Twist。
  4. 200 Hz 计时器 `buffer_update_callback()`：若三个通道齐全则写入 deque。
  5. 键盘 `;` 开始录制、`'` 结束录制，落盘至 `humanoidverse/logs/delta_a_realdata/<exp>/<timestamp>/motion_<id>.npz`。
- **写入内容**：
  - `time`
  - `joint_pos / joint_vel / tau_est`
  - `IMU_quaternion / IMU_gyro / IMU_acc`
  - `joint_pos_cmd / joint_vel_cmd / kp / kd / tau_cmd`
  - `pos / quat / lin_vel / ang_vel`（动捕）
- **使用建议**：
  - 保证 ROS2 与 Unitree SDK 共用时间源，可通过 NTP 同步。
  4. 录制前调用 `ChannelFactoryInitialize` 的 `DOMAIN_ID` 与机器人一致。
  - 若需要额外信息（足底力、扰动标记），可在缓冲结构中扩展字段。

---

## 6. G1 配置要点（`humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml`）
- **DOF 定义**：23 个关节 + 腰部 3 自由度，上肢 8 自由度，与动捕映射直接关联。
- **限制**：`dof_pos_lower/upper`, `dof_vel_limit_list`, `dof_effort_limit_list` 是残差训练的安全边界。
- **控制器**：
  - 基于 P 控制（`control_type: P`）。
  - `action_scale=0.25`，残差策略应依据此比例生成目标角度。
  - `clip_torques=True`，若残差直接输出力矩需考虑饱和影响。
- **初始姿态与随机化**：
  - `init_state.default_joint_angles` 提供站立参考。
  - `randomize_link_body_names` & `extend_config` 在仿真训练时提供随机质量/结构，可在真实数据记录中保存对应参数以便复现。

---

## 7. 训练与部署模板
1. **准备环境**
   ```bash
   conda activate hvgym  # 或 IsaacLab / HVLab 环境
   export UNITREE_SDK_PATH=...</br>
   ```
2. **采集数据**
   ```bash
   cd sim2real
   python rl_policy/listener_deltaa.py --config config/g1.yaml --exp_name walk_mocap
   ```
   - 按 `;` 开始、`'` 结束；数据保存在 `humanoidverse/logs/delta_a_realdata/walk_mocap/<timestamp>/`。
3. **构建残差数据集**
   - 脚本示例：`scripts/data_process/`（可扩展）读取 `.npz` 与动捕元数据，生成 `{state, reference, env} -> residual`。
4. **训练策略**
   ```bash
   python humanoidverse/train_agent.py \
     +simulator=isaacgym \
     +exp=train_delta_a_closed_loop \
     +robot=g1/g1_29dof_anneal_23dof \
     +obs=delta_a/train_policy_with_delta_a \
     data.residual_dataset=/path/to/npz_folder
   ```
   - 根据需要替换奖励、噪声、domain rand。
5. **部署验证**
   - 先在仿真 `humanoidverse/eval_agent.py` 验证。
   - 真实机部署前设置安全限幅（力矩、角度、速度）。

---

## 8. 常见问题
- **Q: 动捕与机器人不同步？**
  - 对齐时间戳，必要时在 `buffer_update_callback` 中记录各源原始时间，训练时进行插值。
- **Q: 数据缺失导致 buffer 不写？**
  - 程序会在缺失信号时通过 ROS warn 提醒，可先排查网络/DDS 配置。
- **Q: 如何加入足底力/外部扰动？**
  - 在 `LowStateHandler` 中扩展字段（如 `foot_force`），同步保存即可。
- **Q: 多执行者动捕如何管理？**
  - 在 `obs` 结构中新增 `metadata`，包含执行者 ID、动作标签、环境参数，方便后续分层抽样。

---

## 9. 参考资料
- `MOCAP_RESIDUAL_LEARNING_MINDMAP.md`：思维导图与采集 checklist。
- Unitree SDK v2 文档：了解 `LowState_ / LowCmd_` 字段释义。
- HumanoidVerse 文档：掌握训练命令与配置体系。

如需进一步扩展（如多机协同、其他机器人），请在此分支基础上继续提交 PR。

---

## 10. 致谢与引用
- **Unitree 官方论文引用**  
  ```
  @article{unitree2024g1,
    title={Unitree G1: General-Purpose Humanoid Platform for Real-World Deployment},
    author={Unitree Robotics},
    journal={Unitree Technical Whitepaper},
    year={2024}
  }
  ```
- **原始 ASAP 仓库**  
  本分支的动捕与残差学习流程深受 [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) 启发，感谢原作者开放代码与数据资料。
