# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.managers import CurriculumTermCfg

from . import mdp

##
# Pre-defined configs
##

from .GR1_T1 import GR1_T1_CFG  # isort:skip


##
# Scene definition
##


@configclass
class WalkSceneCfg(InteractiveSceneCfg):
    """Configuration for a walk scene."""

    # robot
    robot: ArticulationCfg = GR1_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # ground plane
    plane = AssetBaseCfg(
        prim_path = "/World/GroundPlane",
        init_state = AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot",
                                                debug_vis=True,
                                                joint_names=["waist_yaw",
                                                            "l_hip_roll",
                                                            "l_hip_yaw",
                                                            "l_knee_pitch",
                                                            "l_ankle_pitch",
                                                            "l_ankle_roll",
                                                            "r_hip_roll",
                                                            "r_hip_yaw",
                                                            "r_hip_pitch",
                                                            "r_knee_pitch",
                                                            "r_ankle_pitch",
                                                            "r_ankle_roll",
                                                            "waist_pitch",
                                                            "waist_roll",
                                                            "head_yaw",
                                                            "head_roll",
                                                            "head_pitch",
                                                            "l_shoulder_roll",
                                                            "l_shoulder_yaw",
                                                            "l_elbow_pitch",
                                                            "l_wrist_yaw",
                                                            "l_wrist_roll",
                                                            "l_wrist_pitch",
                                                            "r_shoulder_pitch",
                                                            "r_shoulder_roll",
                                                            "r_shoulder_yaw",
                                                            "r_elbow_pitch",
                                                            "r_wrist_yaw",
                                                            "r_wrist_roll",
                                                            "r_wrist_pitch"],
                                                scale=1.0,
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    velocity = mdp.UniformVelocityCommandCfg(asset_name="robot",
                                             heading_command=False,  # let policy freely discover gait patterns and turning strategies
                                             ranges=mdp.UniformVelocityCommandCfg.Ranges(
                                                 lin_vel_x=(-1.2,1.2),  # not fast, lower than 5km/h on fourier's website
                                                 lin_vel_y=(-0.3,0.3),  # lower sidestepping speed
                                                 ang_vel_z=(-1.0,1.0),
                                             ),
                                             resampling_time_range=(5.0,5.0),  # change velocity target every five seconds
                                             debug_vis=True,
                                             )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                               noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))
        base_pos_z = ObsTerm(func=mdp.base_pos_z,
                             noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))

        # joint state
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
                                noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,
                                noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))

        # task command
        command = ObsTerm(func=mdp.generated_commands,
                          params={"command_name":"velocity"})
        
        # previous action
        action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True # allow observations to be "corrupted" by noise
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset,
                                   mode="reset",
                                   params={
                                       "position_range": (0, 0),
                                       "velocity_range": (0, 0),
                                    },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-3.0)
    # (3) Command tracking
    track_lin_xy_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5,
                               params={
                                   "std": 0.25,
                                   "command_name": "velocity",
                                   }
    )
    track_ang_z_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.7,
                              params={"std": 0.25,
                                      "command_name": "velocity",
                                      },
    )
    # (4) Posture/Balance
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0,
                               params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewTerm(func=mdp.base_height_l2,weight=-0.2,
                          params={
                              "target_height": 1.0,
                              "asset_cfg": SceneEntityCfg("robot")},
    )
    # (5) smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)  # penalize sudden changes in actions
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)  # penalize large actions overall
    joint_vel = RewTerm(func=mdp.joint_vel_l2,weight=-0.001,params={"asset_cfg": SceneEntityCfg("robot")},)

    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Bad orientation (fallen down)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, 
                               params={"limit_angle": 0.7,
                                       "asset_cfg": SceneEntityCfg("robot"),
                                       },
    )
    # (3) Base height too low
    base_height = DoneTerm(func=mdp.root_height_below_minimum,
                           params={"minimum_height": 0.6,
                                   "asset_cfg": SceneEntityCfg("robot"),
                                   },
    )
    # (4) Joint position limits violated
    joint_pos_limits = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



##
# Environment configuration
##


@configclass
class WalkEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the walking environment"""
    # Scene settings
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=256, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


@configclass
class WalkEnvCfg_PLAY(WalkEnvCfg):
    """custom environment configuration for playback"""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 4
        # disable randomization for play
        self.observations.policy.enable_corruption = False