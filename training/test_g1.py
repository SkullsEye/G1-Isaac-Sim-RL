import sys
import os

# --- CLEANING ---
sys.path = [p for p in sys.path if "/opt/ros" not in p]
# ----------------

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=False)
simulation_app = launcher.app

import torch
import gymnasium as gym # Imported here so it's defined!
from isaaclab.envs import ManagerBasedRLEnv

# Import your config class directly
from isaaclab_tasks.manager_based.manipulation.pick_place.config_g1_pickplace import PickPlaceG1EnvCfg

def main():
    print("Creating G1 Environment Direct...")
    
    # 1. Instantiate Config
    cfg = PickPlaceG1EnvCfg()
    cfg.scene.num_envs = 1
    
    # 2. Instantiate Environment directly (Bypassing Gym Registry)
    env = ManagerBasedRLEnv(cfg=cfg)
    
    print("Environment Created! Resetting...")
    observation, info = env.reset()
    
    print("Robot is live. Simulating...")
    
    while simulation_app.is_running():
        # Zero actions
        actions = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        env.step(actions)
        
    env.close()

if __name__ == "__main__":
    main()
