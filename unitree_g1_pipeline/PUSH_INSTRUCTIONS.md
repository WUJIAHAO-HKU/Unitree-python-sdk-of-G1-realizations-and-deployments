# Git Push Checklist

1. **Collect files**
   - `docs/` → copy `unitree_g1_pipeline/README.md` (rename to project README if desired).
   - `scripts/` → copy your customized SDK scripts (`test_dds_connection.py`, `get_robot_state.py`, `pd_stand.py`, `deploy_policy.py`).
   - `configs/` → copy Hydra/YAML configs you want to share (`humanoidverse/config/...`).
   - Optional: training shell scripts (`train_big_stride.sh`).

2. **Create repo folder**
   ```bash
   mkdir -p ~/Unitree-python-sdk-of-G1-realizations-and-deployments
   cp -r unitree_g1_pipeline ~/Unitree-python-sdk-of-G1-realizations-and-deployments/docs
   cp unitree_sdk2_python/example/g1/low_level/{test_dds_connection.py,get_robot_state.py,pd_stand.py,deploy_policy.py} \
      ~/Unitree-python-sdk-of-G1-realizations-and-deployments/scripts
   cp train_big_stride.sh ~/Unitree-python-sdk-of-G1-realizations-and-deployments/
   ```

3. **Initialize git & push**
   ```bash
   cd ~/Unitree-python-sdk-of-G1-realizations-and-deployments
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<YOUR_ACCOUNT>/Unitree-python-sdk-of-G1-realizations-and-deployments.git
   git push -u origin main
   ```

4. **Set repo description & topics**
   - Use `unitree_g1_pipeline/GITHUB_DESCRIPTION.md` contents.
   - Upload screenshots / media if available.

5. **Double-check**
   - README renders correctly on GitHub.
   - Scripts have executable instructions.
   - Provide license (e.g., BSD-3-Clause or MIT) depending on Unitree requirements.
