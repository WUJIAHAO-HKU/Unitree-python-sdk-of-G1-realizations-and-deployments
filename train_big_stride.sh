#!/bin/bash

# 大步行走训练脚本
# 无参考轨迹，纯靠奖励函数训练机器人大步向前走

echo "==================================================
大步行走训练脚本 (Big Stride Walking)
==================================================
"

# 训练配置
NUM_ENVS=4096
NUM_ITERATIONS=15000
HEADLESS=True
PROJECT_NAME="BigStrideWalking"
EXPERIMENT_NAME="BigStride_v1"

echo "📋 训练配置:"
echo "  - 环境数量: $NUM_ENVS"
echo "  - 迭代次数: $NUM_ITERATIONS"
echo "  - 无头模式: $HEADLESS"
echo "  - 项目名称: $PROJECT_NAME"
echo "  - 实验名称: $EXPERIMENT_NAME"
echo ""
echo "🎯 训练目标:"
echo "  - 机器人学会大步向前走（无参考轨迹）"
echo "  - 目标步幅: 0.4m"
echo "  - 目标速度: 1.0 m/s"
echo "  - 保持直立稳定"
echo ""
echo "开始训练..."
echo "=================================================="

python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=big_stride_walking \
  +domain_rand=NO_domain_rand \
  +rewards=big_stride/reward_big_stride_walking \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=big_stride/big_stride_obs_basic \
  num_envs=$NUM_ENVS \
  project_name=$PROJECT_NAME \
  experiment_name=$EXPERIMENT_NAME \
  robot.asset.self_collisions=0 \
  algo.config.entropy_coef=0.001 \
  algo.config.init_noise_std=0.5 \
  algo.config.num_learning_iterations=$NUM_ITERATIONS \
  headless=$HEADLESS

echo ""
echo "=================================================="
echo "✅ 训练完成！"
echo "=================================================="
echo "训练信息:"
echo "  - 项目名称: $PROJECT_NAME"
echo "  - 实验名称: $EXPERIMENT_NAME"
echo ""
echo "日志位置:"
echo "  logs/$PROJECT_NAME/<timestamp>-$EXPERIMENT_NAME/"
echo ""
echo "📊 查看训练曲线:"
echo "  tensorboard --logdir logs/$PROJECT_NAME"
echo ""
echo "🎮 测试训练好的模型:"
echo "  python humanoidverse/eval_agent.py \\"
echo "    +checkpoint=logs/$PROJECT_NAME/<timestamp>-$EXPERIMENT_NAME/model_XXXX.pt"
echo ""
echo "💡 提示:"
echo "  - 大步行走通常需要2000-5000次迭代收敛"
echo "  - 观察tensorboard中的 forward_velocity 和 big_stride 奖励"
echo "  - 如果机器人容易摔倒，可以降低 target_forward_velocity"
echo "=================================================="

