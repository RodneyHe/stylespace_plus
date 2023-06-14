#!/usr/bin/env python3
# coding: utf-8

"""
Reference: https://github.com/YadiraF/PRNet/blob/master/utils/estimate_pose.py
"""

import torch
from math import cos, sin, atan2, asin, sqrt
import numpy as np
from .params import param_mean, param_std

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

param_mean = torch.from_numpy(param_mean).to(DEVICE)
param_std = torch.from_numpy(param_std).to(DEVICE)

def parse_pose(params):
    camera_param_list = []
    pose_yaw_list = []
    pose_pitch_list = []
    pose_roll_list = []
    for param in params:
        param = param * param_std + param_mean
        Ps = param[:12].reshape(3, -1)  # camera matrix
        # R = P[:, :3]
        s, R, t3d = P2sRt(Ps)
        P = torch.concat((R, t3d.reshape(3, -1)), axis=1)  # without scale parameters
        camera_param_list.append(P)
        # P = Ps / s
        yaw, pitch, roll = matrix2angle(R)  # yaw, pitch, roll
        # offset = p_[:, -1].reshape(3, 1)
        pose_yaw_list.append(yaw[None, ...])
        pose_pitch_list.append(pitch[None, ...])
        pose_roll_list.append(roll[None, ...])
    pose_yaw = torch.concat(pose_yaw_list, 0)
    pose_pitch = torch.concat(pose_pitch_list, 0)
    pose_roll = torch.concat(pose_roll_list, 0)
    return torch.concat(camera_param_list, 0), torch.concat([pose_yaw, pose_pitch, pose_roll], 0)


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = torch.arcsin(R[2, 0])
        y = torch.arctan2(R[2, 1] / torch.cos(x), R[2, 2] / torch.cos(x))
        z = torch.arctan2(R[1, 0] / torch.cos(x), R[0, 0] / torch.cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = torch.from_numpy(np.pi / 2)
            y = z + torch.arctan2(R[0, 1], R[0, 2])
        else:
            x = torch.from_numpy(-np.pi / 2)
            y = -z + torch.arctan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (torch.linalg.norm(R1) + torch.linalg.norm(R2)) / 2.0
    r1 = R1 / torch.linalg.norm(R1)
    r2 = R2 / torch.linalg.norm(R2)
    r3 = torch.cross(r1, r2)

    R = torch.concat((r1, r2, r3), 0)
    return s, R, t3d


def main():
    pass


if __name__ == '__main__':
    main()
