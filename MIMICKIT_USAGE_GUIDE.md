# MimicKit 训练框架使用指南

## 概述

MimicKit 是一个用于训练运动模仿控制器的强化学习框架。它提供了多种运动模仿方法的实现，包括：
- **DeepMimic**: 基于示例引导的深度强化学习
- **AMP**: 对抗性运动先验
- **ASE**: 大规模可重用对抗技能嵌入
- **ADD**: 对抗差分判别器

支持的强化学习算法：
- **PPO** (Proximal Policy Optimization)
- **AWR** (Advantage-Weighted Regression)

## 安装步骤

### 1. 安装 IsaacGym

首先需要安装 NVIDIA IsaacGym：
- 访问：https://developer.nvidia.com/isaac-gym
- 下载并安装 IsaacGym Preview 4

### 2. 克隆 MimicKit 仓库

```bash
git clone https://github.com/xbpeng/MimicKit.git
cd MimicKit
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载资源和运动数据

从 MimicKit 提供的链接下载资源和运动数据，然后解压到 `data/` 目录下。

## 训练模型

### 基本训练命令

使用命令行参数进行训练：

```bash
python mimickit/run.py \
    --mode train \
    --num_envs 4096 \
    --env_config data/envs/deepmimic_humanoid_env.yaml \
    --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
    --visualize true \
    --log_file output/log.txt \
    --out_model_file output/model.pt
```

### 参数说明

- `--mode`: 选择模式，`train` 或 `test`
- `--num_envs`: 并行环境数量（用于加速训练，建议 4096 或更多）
- `--env_config`: 环境配置文件路径
- `--agent_config`: 智能体配置文件路径
- `--visualize`: 是否启用可视化（训练时建议关闭以加速）
- `--log_file`: 输出日志文件路径
- `--out_model_file`: 输出模型文件路径（保存训练好的模型）
- `--logger`: 日志记录器，可选 `tb` (TensorBoard) 或 `wandb`

### 使用参数文件

也可以使用参数文件来配置训练：

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --visualize true
```

参数文件中的参数与命令行参数等效。所有算法的参数文件都在 `args/` 目录下。

## 测试模型

训练完成后，可以使用以下命令测试模型：

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --num_envs 4 \
    --visualize true \
    --mode test \
    --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```

### 测试参数说明

- `--mode test`: 设置为测试模式
- `--num_envs 4`: 测试时使用较少的并行环境
- `--model_file`: 指定训练好的模型文件路径
- 预训练模型位于 `data/models/` 目录
- 对应的训练日志位于 `data/logs/` 目录

## 分布式训练

如果需要使用多 CPU 或多 GPU 进行分布式训练：

```bash
python mimickit/run.py \
    --arg_file args/deepmimic_humanoid_ppo_args.txt \
    --num_workers 2 \
    --device cuda:0
```

### 分布式训练参数

- `--num_workers`: 用于并行化训练的 worker 进程数量
- `--device`: 训练设备，可以是 `cpu` 或 `cuda:0`
- **注意**: 使用多 GPU 时，worker 进程数量必须小于或等于可用 GPU 数量

## 可视化训练日志

### 使用 TensorBoard

如果训练时使用了 TensorBoard 日志记录器，会在日志文件相同的输出目录下生成 TensorBoard 事件文件。使用以下命令查看：

```bash
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

### 使用绘图脚本

也可以使用提供的绘图脚本 `plot_log.py` 来绘制日志文件：

```bash
python plot_log.py output/log.txt
```

## 运动数据

### 运动数据格式

- 运动数据存储在 `data/motions/` 目录
- 环境配置文件中的 `motion_file` 字段用于指定参考运动片段
- 除了模仿单个运动片段，`motion_file` 也可以指定数据集文件（位于 `data/datasets/`），用于训练模仿多个运动片段的模型

### 运动数据表示

运动片段由 `motion.py` 中实现的 `Motion` 类表示，每个运动片段存储在 `.pkl` 文件中。

每一帧的运动数据格式为：
```
[根位置 (3D), 根旋转 (3D), 关节旋转]
```

其中：
- 3D 旋转使用 3D 指数映射表示
- 1D 关节旋转使用旋转角度表示
- 关节旋转顺序与 `.xml` 文件中的关节顺序一致（深度优先遍历）

### 可视化运动数据

使用 `view_motion` 环境可以可视化运动片段：

```bash
python mimickit/run.py \
    --mode test \
    --arg_file args/view_motion_humanoid_args.txt \
    --visualize true
```

## 运动重定向

如果需要将运动数据重定向到不同的机器人，可以使用 GMR (Generalized Motion Retargeting)。转换脚本位于 `tools/gmr_to_mimickit/`。

## 训练流程总结

1. **准备环境**
   - 安装 IsaacGym
   - 安装依赖包
   - 下载资源和运动数据

2. **配置训练**
   - 选择环境配置文件（`env_config`）
   - 选择智能体配置文件（`agent_config`）
   - 或使用预定义的参数文件（`arg_file`）

3. **开始训练**
   - 设置合适的并行环境数量（`num_envs`）
   - 关闭可视化以加速训练（`visualize false`）
   - 指定输出路径（`log_file`, `out_model_file`）

4. **监控训练**
   - 使用 TensorBoard 或 wandb 查看训练曲线
   - 定期检查日志文件

5. **测试模型**
   - 使用训练好的模型进行测试
   - 启用可视化查看效果

## 常见问题

### 训练速度慢
- 增加 `num_envs` 参数（如果 GPU 内存允许）
- 设置 `visualize false` 关闭可视化
- 使用分布式训练（多 GPU）

### 内存不足
- 减少 `num_envs` 参数
- 使用 CPU 训练（`device cpu`）

### 模型不收敛
- 检查运动数据是否正确加载
- 调整奖励函数参数
- 查看训练日志中的奖励曲线

## 引用

如果使用 MimicKit，请引用：

```bibtex
@misc{
    MimicKitPeng2025,
    title={MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control}, 
    author={Xue Bin Peng},
    year={2025},
    eprint={2510.13794},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2510.13794}, 
}
```

## 相关资源

- GitHub 仓库: https://github.com/xbpeng/MimicKit
- Starter Guide: 查看仓库中的详细文档
- 预训练模型: `data/models/` 目录
- 示例配置: `args/` 目录

