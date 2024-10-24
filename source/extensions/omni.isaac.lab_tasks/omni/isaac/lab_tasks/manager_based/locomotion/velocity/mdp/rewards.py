# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat,quat_rotate

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    term = env.action_manager.get_term("joint_pos_pmtg")
    threshold = term.pmtg.desire_feet_air_time
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1).clip(max=0.1)
    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stumble(env: ManagerBasedRLEnv,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w
    # print(torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=2).cpu().numpy())
    return torch.any(torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=2) > \
                         5 * torch.abs(net_contact_forces[:, sensor_cfg.body_ids, 2]), dim=1)

def feet_height(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    pmtg_term = env.action_manager.get_term("joint_pos_pmtg")
    diff_in_world = (asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1).repeat(1, 4, 1)).reshape([-1,3])
    body_quat_w = asset.data.body_quat_w[:, 0, :].unsqueeze(1).repeat(1,4,1).reshape([-1,4])
    feet_pos_in_base = quat_rotate_inverse(body_quat_w,diff_in_world).reshape([env.num_envs, 4, 3])
    error = feet_pos_in_base - pmtg_term.pmtg.foot_target_position_in_base_frame[:, :, :]
    return torch.sum(error ** 2, dim=(1, 2))

def feet_xyz(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def delta_phi(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    pmtg_term = env.action_manager.get_term("joint_pos_pmtg")
    delta_phi_norm = torch.norm(pmtg_term.delta_phi, dim=-1)
    return delta_phi_norm

def residual_angle(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    base_quat = base_quat[:, [1, 2, 3, 0]]

    pmtg_term = env.action_manager.get_term("joint_pos_pmtg")
    origin_angles = pmtg_term.pmtg.get_origin_action(base_quat)
    residual_angle_norm = torch.norm((origin_angles - asset.data.joint_pos), dim=-1)
    return residual_angle_norm

def target_smoothness(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    pmtg_term = env.action_manager.get_term("joint_pos_pmtg")
    # Penalize changes in actions
    return torch.sum(torch.square(pmtg_term.last_processed_actions - pmtg_term.processed_actions) +
                        torch.square(pmtg_term.processed_actions - 2 * pmtg_term.last_processed_actions + pmtg_term.last_last_processed_actions),
                        dim=1)