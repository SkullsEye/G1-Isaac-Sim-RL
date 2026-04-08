# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym
import os
import yaml
from dataclasses import dataclass, asdict
from . import move_cylinder_g1_29dof_dex1_hw_env_cfg

# --- CONFIG CLASSES ---
@dataclass
class PpoAlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    
    def to_dict(self):
        return asdict(self)
        
    def from_dict(self, cfg_dict):
        for key, value in cfg_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class ActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    activation: str = "elu"
    
    def to_dict(self):
        return asdict(self)

    def from_dict(self, cfg_dict):
        for key, value in cfg_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class G1PPORunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = "g1_move_cylinder"
    run_name: str = ""
    logger: str = "tensorboard"
    resume: bool = False
    load_run: int = -1
    load_checkpoint: int = -1
    # --- ADDED MISSING FLAGS ---
    clip_actions: float = 1.0 
    empirical_normalization: bool = False
    # ---------------------------
    algorithm: PpoAlgorithmCfg = None
    policy: ActorCriticCfg = None
    
    def to_dict(self):
        return asdict(self)

    def from_dict(self, cfg_dict):
        # Update top-level attributes
        for key, value in cfg_dict.items():
            if key == "algorithm" and isinstance(value, dict) and self.algorithm:
                self.algorithm.from_dict(value)
            elif key == "policy" and isinstance(value, dict) and self.policy:
                self.policy.from_dict(value)
            elif hasattr(self, key):
                setattr(self, key, value)

# --- LOADER FUNCTION ---
def load_g1_ppo_cfg():
    # 1. Path to YAML
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "g1_ppo.yaml")
    
    # 2. Load Dictionary using Standard YAML
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # 3. Convert Dict to Object
    runner_dict = cfg_dict["runner"]
    algo_dict = runner_dict["algorithm"]
    policy_dict = runner_dict["policy"]

    cfg = G1PPORunnerCfg()
    # Copy values from YAML to Object
    for key, value in runner_dict.items():
        if hasattr(cfg, key) and key not in ["algorithm", "policy"]:
            setattr(cfg, key, value)
            
    cfg.algorithm = PpoAlgorithmCfg(**{k: v for k, v in algo_dict.items() if k != "class_name"})
    cfg.policy = ActorCriticCfg(**{k: v for k, v in policy_dict.items() if k != "class_name"})
    
    return cfg

# --- REGISTRATION ---
gym.register(
    id="Isaac-Move-Cylinder-G129-Dex1-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_cylinder_g1_29dof_dex1_hw_env_cfg.MoveCylinderG129Dex1WholebodyEnvCfg,
        "rsl_rl_cfg_entry_point": load_g1_ppo_cfg, 
    },
    disable_env_checker=True,
)
