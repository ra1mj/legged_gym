# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2025 ra1mj
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from base.legged_robot_config import LeggedRobotCfg



def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class nova_env(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        # if hasattr(self, "_custom_init"):
        #     self._custom_init(cfg)
        self.init_done = True

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs


#----env_init-----------#




#----create_robot-------#




#----explore------------#




#----progress-----------#



#----reward-------------#


#----reward_func--------#
    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return 0.5 + 0.5 * torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])

    def _reward_tracking_base_height(self):
        # Penalize base height away from target
        base_height_error = torch.square(self.base_pos[:, 2] - self.commands[:, 3])
        return torch.exp(-base_height_error / self.reward_cfg["tracking_height_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_joint_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,0:4] - self.actions[:,0:4]), dim=1)

    def _reward_wheel_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,4:6] - self.actions[:,4:6]), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        #个人认为为了灵活性这个作用性不大
        return torch.sum(torch.abs(self.dof_pos[:,0:4] - self.default_dof_pos[:,0:4]), dim=1)

    def _reward_projected_gravity(self):
        #保持水平奖励使用重力投影 0 0 -1
        #使用e^(-x^2)效果不是很好
        projected_gravity_error = 1 + self.projected_gravity[:, 2] #[0, 0.2]
        projected_gravity_error = torch.square(projected_gravity_error)
        # projected_gravity_error = torch.square(self.projected_gravity[:,2])
        return torch.exp(-projected_gravity_error / self.reward_cfg["tracking_gravity_sigma"])
        # return torch.sum(projected_gravity_error)

    def _reward_similar_legged(self):
        # 两侧腿相似 适合使用轮子行走 抑制劈岔，要和_reward_knee_height结合使用
        legged_error = torch.sum(torch.square(self.dof_pos[:,0:2] - self.dof_pos[:,2:4]), dim=1)
        return torch.exp(-legged_error / self.reward_cfg["tracking_similar_legged_sigma"])
        # return legged_error

    def _reward_knee_height(self):
        # 关节处于某个范围惩罚，避免总跪着
        left_knee_idx = torch.abs(self.left_knee_pos[:, 2]) < 0.08
        right_knee_idx = torch.abs(self.right_knee_pos[:, 2]) < 0.08
        knee_rew = torch.sum(torch.square(self.left_knee_pos[left_knee_idx, 2] - 0.08)) if left_knee_idx.any() else 0
        knee_rew += torch.sum(torch.square(self.right_knee_pos[right_knee_idx, 2] - 0.08)) if right_knee_idx.any() else 0
        return knee_rew


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :4]), dim=1)

    def _reward_dof_acc(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.dt))

    def _reward_dof_force(self):
        # Penalize z axis base linear velocity
        return torch.sum(torch.square(self.dof_force), dim=1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_collision(self):
        # 接触地面惩罚 力越大惩罚越大
        collision = torch.zeros(self.num_envs,device=self.device,dtype=gs.tc_float)
        for idx in self.reset_links:
            collision += torch.square(self.connect_force[:,idx,:]).sum(dim=1)
        return collision



