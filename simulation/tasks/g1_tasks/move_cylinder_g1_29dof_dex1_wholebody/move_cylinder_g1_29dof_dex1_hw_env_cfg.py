# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes import CylinderCfg
# FIXED: Import Actuator Config correctly
from isaaclab.actuators import ImplicitActuatorCfg
from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets

# --- PATHS ---
UNITREE_ASSETS_PATH = "/home/umz/unitree_sim_isaaclab/assets"

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Object table scene configuration."""
    
    # 1. Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
    )
    
    # 2. Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 3. Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UsdFileCfg(
            usd_path=f"{UNITREE_ASSETS_PATH}/objects/PackingTable/PackingTable.usd", 
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.0, 0.0), rot=(0.707, 0, 0, 0.707)),
    )
    
    # 4. Object
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=CylinderCfg(
            radius=0.03,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 1.05)), 
    )

    # 5. Humanoid robot (G1) with CORRECT STIFFNESS
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(
        init_pos=(0.0, 0.0, 0.82), 
        init_rot=(1, 0, 0, 0),
    )
    
    # OVERRIDE ACTUATORS (Manual Stiffening)
    # Corrected "torso" to "waist" based on your logs
    robot.actuators["legs_stiff"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
        stiffness=200.0,
        damping=5.0,
    )
    robot.actuators["upper_body_stiff"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", "waist_.*"], # FIXED: waist instead of torso
        stiffness=100.0,
        damping=5.0,
    )

    # Sensors
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=10, track_air_time=True, debug_vis=False)
    front_camera = CameraPresets.g1_front_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action configuration."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation configuration."""
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gripper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)
        object_position = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Termination configuration."""
    pass

@configclass
class RewardsCfg:
    """Reward configuration."""
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)

@configclass
class EventCfg:
    """Configuration for events (Resets)."""
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
    )

@configclass
class MoveCylinderG129Dex1WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """Main environment configuration."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
