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
)