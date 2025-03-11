import copy
import torch
import numpy as np
import mujoco
import torch
import torch.utils.data as data
import numpy as np

import pathlib
from math import ceil
from typing import Any, List, Tuple


class DummyCam:
    def __init__(
        self, azimuth: float, elevation: float, distance: float, lookat=[0, 0, 0]
    ):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = copy.deepcopy(lookat)


def generate_camera(dummy: DummyCam) -> mujoco.MjvCamera:
    # Configure the camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = dummy.distance
    cam.azimuth = dummy.azimuth
    cam.elevation = dummy.elevation
    cam.lookat = np.asarray(dummy.lookat)

    return cam


def get_norm_func(low, hight):
    lower_limits = np.asarray(low)
    upper_limits = np.asarray(hight)

    scale = 2 / (upper_limits - lower_limits)

    def normalized(joint_positions):
        return (joint_positions - lower_limits) * scale - 1.0

    def unnormalized(joint_positions):
        return (joint_positions + 1.0) / scale + lower_limits

    return normalized, unnormalized


def get_norm_func_torch(low, hight, device="cuda"):
    lower_limits = torch.as_tensor(low, device=device)
    upper_limits = torch.as_tensor(hight, device=device)

    scale = 2 / (upper_limits - lower_limits)

    def normalized(joint_positions):
        return (joint_positions - lower_limits) * scale - 1.0

    def unnormalized(joint_positions):
        return (joint_positions + 1.0) / scale + lower_limits

    return normalized, unnormalized


class JointSubsetData(data.Dataset):
    def __init__(
        self, joint: torch.Tensor, label: torch.Tensor, batch_size=100_000
    ) -> None:
        self.joints = joint.float().cuda().contiguous()
        self.labels = label.float().cuda().contiguous()

        self.batch_size = batch_size

        self._current_idx = 0

    def shuffle(self):
        reindex = torch.randperm(len(self.labels))

        self.joints = self.joints[reindex]
        self.labels = self.labels[reindex]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        index %= len(self)

        first_i, last_i = self.batch_size * index, min(
            self.batch_size * (index + 1), len(self.labels)
        )

        return self.joints[first_i:last_i], self.labels[first_i:last_i]

    def __len__(self):
        return ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx > len(self):
            raise StopIteration

        result = self[self._current_idx]

        self._current_idx += 1

        return result


class DrRobotData(data.Dataset):
    def __init__(self, data_path, t="label", shuffle=True) -> None:
        self.data_path = pathlib.Path(data_path)

        if t == "label":
            joints, gts = DrRobotData.load_label(data_path / "joint_data.npz")
        elif t == "distance":
            joints, gts = DrRobotData.load_distance(data_path / "dist_data.npz")
        elif t == "ctrl":
            joints, gts = DrRobotData.load_ctrl(data_path / "ctrl_data.npz")
        else:
            raise NotImplementedError(t)

        if shuffle:
            reindex = torch.randperm(len(gts))
        else:
            reindex = torch.arange(len(gts))

        self.joints = torch.as_tensor(joints[reindex])
        self.gts = torch.as_tensor(gts[reindex])

    @staticmethod
    def load_ctrl(data_path):
        data = np.load(data_path)
        qpos = data.get("qpos")
        ctrl = data.get("ctrl")
        return qpos, ctrl

    @staticmethod
    def load_label(data_path):
        data = np.load(data_path)
        joint = data.get("joint")
        label = data.get("nums") > 0
        return joint, label[..., np.newaxis]

    @staticmethod
    def load_distance(data_path):
        data = np.load(data_path)
        joint = data.get("joint")
        distance = data.get("dist")
        return joint, distance

    def get_split(
        self, split_ratio=[0.9, 0.1], batchsize=1_000_000
    ) -> List[JointSubsetData]:
        assert sum(split_ratio) == 1
        subset = []
        split_ratio.insert(0, 0.0)
        cumsum = np.cumsum(split_ratio)
        for prev_f, next_f in zip(cumsum[:-1], cumsum[1:]):
            prev_i, next_i = ceil(len(self) * prev_f), ceil(len(self) * next_f)
            subset.append(
                JointSubsetData(
                    self.joints[prev_i:next_i], self.gts[prev_i:next_i], batchsize
                )
            )
        return subset

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.joints[index : index + self.batch_size],
            self.gts[index : index + self.batch_size],
        )

    def __len__(self):
        return len(self.joints)

def traj2order_loss(x, s, t, sdf):
    v = x[1:]-x[:-1]
    a = v[1:]-v[:-1]

    return v.norm(dim=-1).mean() + a.norm(dim=-1).mean() + (x[0]-s).norm() + (x[-1]-t).norm() + (sdf + 0.10).abs().mean()
