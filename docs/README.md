# Unitree G1 Python SDK Full Pipeline

This folder summarizes the end-to-end workflow we verified on the Unitree G1 robot:

1. **Build dependencies** (Cyclone DDS v0.10.x, Unitree SDK2 Python).
2. **Prepare the Python 3.10 environment** dedicated to the SDK.
3. **Verify DDS connectivity** with the robot (`test_dds_connection.py`).
4. **Inspect robot state streams** (`get_robot_state.py`).
5. **Bring the robot up with pure PD control** (`pd_stand.py`).
6. **Deploy the trained RL locomotion policy** (`deploy_policy.py`).
7. **Train / evaluate policies** using the HumanoidVerse configs (e.g., big-stride walking).

The goal is to make it trivial to replicate the "simulation → real" path and to publish the entire process to GitHub.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `unitree_sdk2_python/` | Vendor SDK (editable install). Contains the custom scripts under `example/g1/low_level/`. |
| `humanoidverse/` | Training framework (IsaacGym-based) used to produce the RL policies. |
| `unitree_g1_pipeline/` | **This** folder with documentation & release assets. |

Important scripts:

- `unitree_sdk2_python/example/g1/low_level/test_dds_connection.py`
- `unitree_sdk2_python/example/g1/low_level/get_robot_state.py`
- `unitree_sdk2_python/example/g1/low_level/pd_stand.py`
- `unitree_sdk2_python/example/g1/low_level/deploy_policy.py`
- `humanoidverse/train_agent.py` + `train_big_stride.sh`

## 1. Prerequisites

- Ubuntu 22.04 (Jammy)
- Python 3.10 (conda/venv or system install)
- CUDA-compatible GPU for training (optional for deployment)
- Unitree G1 robot with SDK2 license

### Cyclone DDS

```bash
cd ~/ASAP
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

### Unitree SDK2 Python (editable)

```bash
cd ~/ASAP
/usr/bin/python3.10 -m venv unitree_310_env
source unitree_310_env/bin/activate
pip install -U pip
export CYCLONEDDS_HOME="$HOME/ASAP/cyclonedds/install"
pip install -e unitree_sdk2_python
```

### HumanoidVerse dependencies

Follow the existing instructions in `humanoidverse/README`. The most common training command is already wrapped in `train_big_stride.sh`.

## 2. DDS & Robot Bring-up

1. **Connectivity test**
   ```bash
   cd unitree_sdk2_python/example/g1/low_level
   python test_dds_connection.py <net_iface>
   ```
   - Confirms that DDS transports, `rt/lowstate`, and `rt/lowcmd` are alive.

2. **Real-time state monitor**
   ```bash
   python get_robot_state.py <net_iface>
   ```
   - Streams IMU, joint angles/velocities/torques.

3. **PD stand**
   ```bash
   python pd_stand.py <net_iface>
   ```
   - Locks the G1 into the default 23-DoF stance using the same PD gains as training.

4. **Policy deployment**
   ```bash
   python deploy_policy.py <checkpoint_path> <net_iface>
   ```
   - Loads a PPO actor (HumanoidVerse) and sends actions at 100 Hz.

## 3. Training & Evaluation

- Launch big-stride walking PPO training:
  ```bash
  cd ~/ASAP
  bash train_big_stride.sh
  ```
- Evaluate / export checkpoints:
  ```bash
  python humanoidverse/eval_agent.py +checkpoint=logs/BigStrideWalking/.../model_XXXX.pt
  ```
  This script can export ONNX or TorchScript policies for deployment.

## 4. Recommended README Structure (for the GitHub repo)

When publishing, the top-level README should contain:

1. **Project Title & Badge**
2. **Motivation** – "Unitree G1 full-stack pipeline from IsaacGym PPO training to real hardware"
3. **Quick Start** – the commands shown above
4. **Repository Map** – table similar to this document
5. **Deployment Flowchart** – bullet list: build deps → DDS test → PD stand → Policy deploy
6. **Safety Notes** – ensure remote E-stop ready, start with PD stand, etc.
7. **License / Citation** – credit Unitree SDK2 & HumanoidVerse

## 5. Suggested GitHub Description & Topics

```
Description: End-to-end Unitree G1 locomotion pipeline — cyclonedds setup, DDS diagnostics, PD stand-up, and PPO policy deployment built on Unitree SDK2 Python + HumanoidVerse.

Topics: unitree, sdk2, g1, robotics, reinforcement-learning, isaacgym, deployment, cyclonedds
```

## 6. How to Package & Push

1. Create a clean folder for the GitHub repo (e.g., `Unitree-python-sdk-of-G1-realizations-and-deployments`).
2. Copy the curated files:
   - `unitree_g1_pipeline/README.md` (this file)
   - `unitree_sdk2_python/example/g1/low_level/*.py` scripts you customized
   - `train_big_stride.sh` and relevant configs from `humanoidverse/config`
3. Initialize git & push:
   ```bash
   cd ~/ASAP
   mkdir Unitree-python-sdk-of-G1-realizations-and-deployments
   # copy the files into the new folder
   cp -r unitree_g1_pipeline Unitree-python-sdk-of-G1-realizations-and-deployments/docs
   cp unitree_sdk2_python/example/g1/low_level/{test_dds_connection.py,get_robot_state.py,pd_stand.py,deploy_policy.py} Unitree-python-sdk-of-G1-realizations-and-deployments/scripts
   cp train_big_stride.sh Unitree-python-sdk-of-G1-realizations-and-deployments/
   # add any config yaml you need, e.g.
   cp -r humanoidverse/config/obs/big_stride Unitree-python-sdk-of-G1-realizations-and-deployments/config/obs/

   cd Unitree-python-sdk-of-G1-realizations-and-deployments
   git init
   git add .
   git commit -m "Initial release: Unitree G1 full pipeline"
   git branch -M main
   git remote add origin https://github.com/<YOUR_ORG>/Unitree-python-sdk-of-G1-realizations-and-deployments.git
   git push -u origin main
   ```

## 7. Next Steps

- Automate environment creation (add `environment.yml`).
- Add ROS 2 bridge nodes if needed (e.g., publish `LowState` to ROS topics).
- Provide demo videos / logs inside a `media/` folder in the GitHub repo.

Feel free to adapt this README template when publishing; everything above is safe to copy into the new repository.
