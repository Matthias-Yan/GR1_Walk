# articulation cfg

"""
Configuration for the GR1_T1 robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

GR1_T1_CFG = ArticulationCfg(
    # Where is the USD file
    spawn = sim_utils.UsdFileCfg(
        usd_path = f"/home/matthias/GR1-Walk/GR1_T1.usd",
        # Use default rigid_props and articulation_props in USD file.
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.949), # position, z-axis tested in isaac-sim
        rot = (1.0, 0.0, 0.0 ,0.0), # quaternions, no rotation
        lin_vel = (0.0, 0.0, 0.0),
        ang_vel = (0.0, 0.0, 0.0),
        joint_pos = {},
        joint_vel = {},
    ),
    actuators = {
    "torso": ImplicitActuatorCfg(
        joint_names=[
            "waist_yaw",
            "waist_pitch",
            "waist_roll",
        ],
        effort_limit_sim=120.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),

    "left_leg": ImplicitActuatorCfg(
        joint_names=[
            "l_hip_roll",
            "l_hip_yaw",
            "l_knee_pitch",
            "l_ankle_pitch",
            "l_ankle_roll",
        ],
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),

    "right_leg": ImplicitActuatorCfg(
        joint_names=[
            "r_hip_roll",
            "r_hip_yaw",
            "r_hip_pitch",
            "r_knee_pitch",
            "r_ankle_pitch",
            "r_ankle_roll",
        ],
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),

    "head": ImplicitActuatorCfg(
        joint_names=[
            "head_yaw",
            "head_roll",
            "head_pitch",
        ],
        effort_limit_sim=80.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),

    "left_arm": ImplicitActuatorCfg(
        joint_names=[
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow_pitch",
            "l_wrist_yaw",
            "l_wrist_roll",
            "l_wrist_pitch",
        ],
        effort_limit_sim=120.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),

    "right_arm": ImplicitActuatorCfg(
        joint_names=[
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_shoulder_yaw",
            "r_elbow_pitch",
            "r_wrist_yaw",
            "r_wrist_roll",
            "r_wrist_pitch",
        ],
        effort_limit_sim=120.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
    ),
}
)
