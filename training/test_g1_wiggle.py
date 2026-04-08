import sys
import os
import math

# --- CLEANING ---
# Ensure ROS is stripped from path
sys.path = [p for p in sys.path if "/opt/ros" not in p]
# ----------------

from isaaclab.app import AppLauncher
# Set headless=False so you can see the window
launcher = AppLauncher(headless=False)
simulation_app = launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.pick_place.config_g1_pickplace import PickPlaceG1EnvCfg

def main():
    print("Creating G1 Wiggle Environment (Fixed Base)...")
    cfg = PickPlaceG1EnvCfg()
    cfg.scene.num_envs = 1
    
    # Create Env
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    
    print("Robot is live. Starting Wiggle Test...")
    
    step_count = 0
    while simulation_app.is_running():
        
        # Action Structure: 
        # [Left Arm (6), Right Arm (6), Fingers (8), Legs (~17)]
        # Total size depends on the robot, but we can safely zero-init the full tensor
        actions = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        
        # Create a sine wave for smooth movement
        wiggle = 0.5 * math.sin(step_count * 0.05) 
        
        # --- COMMANDS ---
        # The actions are flattened. Based on config order:
        # 0-5: Left Arm (Differential IK Pose Delta)
        # 6-11: Right Arm (Differential IK Pose Delta)
        
        # Move Left Hand UP/DOWN (Z is index 2)
        actions[:, 2] = wiggle 
        
        # Move Right Hand FORWARD/BACK (X is index 6)
        actions[:, 6] = wiggle

        # LEGS: We leave the rest as 0.0
        # Since we set "use_default_offset=True" in the config, 
        # 0.0 means "Stay in default standing pose"

        # Apply action
        env.step(actions)
        step_count += 1
        
    env.close()

if __name__ == "__main__":
    main()
