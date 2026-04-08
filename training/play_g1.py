# play_g1.py

import argparse
from isaaclab.app import AppLauncher

# 1. LAUNCH SIMULATOR
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 2. IMPORTS
import os
import torch
import glob
from rsl_rl.runners import OnPolicyRunner

# Import Configs
from isaaclab_tasks.manager_based.manipulation.pick_place.config_g1_pickplace import PickPlaceG1EnvCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import Wrapper
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

def get_latest_log_dir(root_dir="logs/g1_pickplace"):
    """Finds the most recently modified log folder."""
    if not os.path.exists(root_dir):
        print(f"[ERROR] Could not find log directory: {root_dir}")
        return None
    subdirs = glob.glob(os.path.join(root_dir, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if not subdirs:
        print(f"[ERROR] No folders found in {root_dir}")
        return None
    return max(subdirs, key=os.path.getmtime)

# --- 3. RECONSTRUCT THE BRAIN CONFIG (COMPLETE) ---
agent_cfg = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 5000,
    "save_interval": 50,
    "experiment_name": "g1_pickplace",
    "run_name": "",
    "resume": True,
    "load_run": -1,
    "checkpoint": -1,
    "empirical_normalization": False,
    "logger": "tensorboard",

    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0
    },
    "policy": {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 1.0
    },
}

def main():
    # Find latest log
    log_dir = get_latest_log_dir()
    if not log_dir:
        return
    print(f"\n[INFO] Loading experiment from: {log_dir}")

    # Setup Environment
    env_cfg = PickPlaceG1EnvCfg()
    env_cfg.scene.num_envs = 1 
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # WRAP ENVIRONMENT
    env = RslRlVecEnvWrapper(env)

    # Load Policy
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device="cuda:0")
    
    # --- FIX IS HERE ---
    # We explicitly load the best model found in the folder
    model_path = os.path.join(log_dir, "model_1250.pt") # Fallback guess
    
    # Try to find the latest model file automatically
    model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if model_files:
        # Sort to find the highest number (latest checkpoint)
        model_path = max(model_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        print(f"[INFO] Auto-loading checkpoint: {model_path}")
        runner.load(model_path)
    else:
        print("[WARNING] No model_*.pt found! Starting with random weights.")

    policy = runner.get_inference_policy(device="cuda:0")

    # Play Loop
    obs, _ = env.get_observations()
    print("\n>>> PLAYING POLICY... Press Ctrl+C to stop.\n")
    
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        
        obs, _, _, _ = env.step(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()
