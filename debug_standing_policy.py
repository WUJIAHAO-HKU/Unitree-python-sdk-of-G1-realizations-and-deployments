#!/usr/bin/env python3
"""
Debug script to analyze standing policy behavior
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Load ONNX model
onnx_path = Path("/home/wujiahao/ASAP/logs/Standing/20251017_162813-My_Standing_Task-standing-g1_29dof_anneal_23dof/exported/model_700.onnx")
session = ort.InferenceSession(str(onnx_path))

# Test with different observation scenarios
def test_observation(name, obs):
    """Test policy response to specific observation"""
    obs = obs.reshape(1, -1).astype(np.float32)
    action = session.run(None, {session.get_inputs()[0].name: obs})[0]
    action = action.flatten()
    
    print(f"\n=== {name} ===")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"Action mean: {action.mean():.3f}, std: {action.std():.3f}")
    print(f"Max abs action: {np.abs(action).max():.3f}")
    
    return action

# Create test observations (80-dim)
# Format: [dof_pos(23), dof_vel(23), actions(23), base_lin_vel(3), base_ang_vel(3), projected_gravity(3), target_height(1), feet_distance(1)]

print("Testing standing policy with different scenarios...")

# 1. Perfect standing pose (all zeros except gravity and target values)
obs_perfect = np.zeros(80)
obs_perfect[69:72] = [0, 0, 1]  # projected_gravity: [0, 0, 1] (upright)
obs_perfect[72] = 0.78  # target_height
obs_perfect[73] = 0.3   # feet_distance
action_perfect = test_observation("Perfect Standing", obs_perfect)

# 2. Slightly tilted forward
obs_tilted = obs_perfect.copy()
obs_tilted[69:72] = [0.1, 0, 0.995]  # slight forward tilt
action_tilted = test_observation("Tilted Forward", obs_tilted)

# 3. Default joint positions from config
default_joints = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg  
    0.0, 0.0, 0.0,                   # waist
    0.0, 0.0, 0.0, 0.0,              # left arm
    0.0, 0.0, 0.0, 0.0               # right arm
])

obs_default = obs_perfect.copy()
obs_default[0:23] = default_joints
action_default = test_observation("Default Joints", obs_default)

# 4. Test with small joint velocities
obs_moving = obs_default.copy()
obs_moving[23:46] = np.random.normal(0, 0.1, 23)  # small random velocities
action_moving = test_observation("Small Velocities", obs_moving)

# 5. Test action interpretation
print(f"\n=== Action Analysis ===")
print(f"Are actions joint targets? Check if they're close to default_joints:")
print(f"Default joints: {default_joints}")
print(f"Perfect action: {action_perfect}")
print(f"Difference: {action_perfect - default_joints}")

# Check if actions are deltas or absolute targets
if np.abs(action_perfect - default_joints).max() < 0.5:
    print("✓ Actions appear to be ABSOLUTE joint targets")
else:
    print("✓ Actions appear to be joint DELTAS")

print(f"\nRecommended deployment settings:")
if np.abs(action_perfect).max() > 2.0:
    print("- Use --unnormalized-actions")
    print(f"- Use --action-clip-unnormalized {min(0.5, np.abs(action_perfect).max()):.2f}")
else:
    print("- Use normalized actions (default)")
    print(f"- Action scale should be around {0.25}")

print(f"- PD gains: try --pd-scale 0.5 to 1.5")
print(f"- If robot collapses: increase --pd-scale")  
print(f"- If robot oscillates: decrease --pd-scale")
