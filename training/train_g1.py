import sys
import os
import argparse
from datetime import datetime

# --- CLEANING PATH ---
sys.path = [p for p in sys.path if "/opt/ros" not in p]
# ---------------------

from isaaclab.app import AppLauncher

# 1. Parse Arguments
parser = argparse.ArgumentParser(description="Train G1 Robot")
parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI")
parser.add_argument("--num_envs", type=int, default=64, help="Number of robots")
args = parser.parse_args()

# 2. Launch Simulator
launcher = AppLauncher(headless=args.headless)
simulation_app = launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.pick_place.config_g1_pickplace import PickPlaceG1EnvCfg

# --- WRAPPER & RUNNER ---
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper 
from rsl_rl.runners import OnPolicyRunner
# We must import these so 'eval' finds them!
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

def main():
    print(f"Initializing G1 Training (Headless: {args.headless})...")
    
    # 3. Setup Config
    env_cfg = PickPlaceG1EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # 4. Create Base Environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # --- WRAP ENVIRONMENT ---
    env = RslRlVecEnvWrapper(env)
    
    # 5. Agent Configuration
    agent_cfg = {
        "empirical_normalization": False,
        "seed": 42,
        "device": env.device,
        "num_steps_per_env": 24,
        "max_iterations": 5000, 
        "save_interval": 100,
        "experiment_name": "g1_pickplace",
        "run_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "logger": "tensorboard",
        
        # --- FIX 1: Add class_name to Policy ---
        "policy": {
            "class_name": "ActorCritic", # <--- CRITICAL FIX
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        
        # --- FIX 2: Ensure Algorithm class is set ---
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        }
    }
    
    # 6. Create Runner
    log_dir = os.path.join("logs", agent_cfg["experiment_name"], agent_cfg["run_name"])
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=env.device)
    
    print(f"Starting training! Logs: {log_dir}")
    
    # 7. Train
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
    
    print("Training Complete.")
    env.close()

if __name__ == "__main__":
    main()
