# debug_g1.py

from isaaclab.app import AppLauncher

# 1. Launch the Simulator
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
# Import your specific G1 config
from isaaclab_tasks.manager_based.manipulation.pick_place.config_g1_pickplace import PickPlaceG1EnvCfg

def main():
    # 2. Load Configuration
    env_cfg = PickPlaceG1EnvCfg()
    env_cfg.scene.num_envs = 1  # Spawn only 1 robot
    
    # 3. Setup Environment (Uses default device, usually GPU)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    print("[INFO] Simulation running. Robot is holding its start pose.")
    print("[INFO] Press Ctrl+C to exit.")
    
    while simulation_app.is_running():
        # 4. Send Zero Actions (Keep robot in default pose)
        zero_actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        env.step(zero_actions)

    simulation_app.close()

if __name__ == "__main__":
    main()
