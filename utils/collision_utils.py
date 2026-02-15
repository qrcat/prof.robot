import copy
import torch
import numpy as np
import mujoco
import torch
import torch.utils.data as data
import numpy as np

import pathlib
from tqdm import tqdm
from math import ceil
from typing import Any, List, Tuple, Union
from plyfile import PlyData, PlyElement


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


class DrDynamicRobotData(data.Dataset):
    """Dynamic Robot Collision Dataset with obstacle point cloud sampling.

    This class organizes data by scene subfolders (sample_XXXXX).
    All scene datasets are pre-loaded and stored for efficient access.
    """

    def __init__(self, data_path, t="label", shuffle=True, num_points=100) -> None:
        """
        Initialize the dataset with multiple scenes and pre-load all scene datasets.

        Args:
            data_path: Path to the dataset directory containing scene folders (sample_XXXXX)
            t: Type of data to load ('label' or 'distance')
            shuffle: Whether to shuffle the data
            num_points: Number of points to sample from obstacle point cloud for each sample
        """
        self.data_path = pathlib.Path(data_path)
        self.num_points = num_points
        self.t = t
        self.shuffle = shuffle

        # Load all scene directories
        scene_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir() and d.name.startswith('sample_')])
        print(f"Found {len(scene_dirs)} scenes")

        # Pre-load and store all scene datasets
        self.scenes = []
        for scene_dir in tqdm(scene_dirs, "Loading datasets..."):
            scene_dataset = DrDynamicSceneDataset(scene_dir, t=t, shuffle=False, num_points=num_points)
            self.scenes.append(scene_dataset)
        
        print(f"Total samples: {len(self)}")

    def collect_all(self):
        all_joints = []
        all_gts = []
        all_pcs = []

        for scene in tqdm(self.scenes, desc="collect all data"):
            joints, gt, pc = scene[:]
            all_joints.append(joints)
            all_gts.append(gt)
            all_pcs.append(pc)
        
        all_joints = torch.cat(all_joints)
        all_gts = torch.cat(all_gts)
        all_pcs = torch.cat(all_pcs)
        
        return all_joints, all_gts, all_pcs

    def get_split(
        self, split_ratio=[0.9, 0.1], batchsize=1_000_000
    ) -> List['JointSubsetDataWithPC']:
        """
        Split dataset into train/test subsets.

        Args:
            split_ratio: Ratio for splitting (e.g., [0.9, 0.1] for 90% train, 10% test)
            batchsize: Batch size for data loading

        Returns:
            List of JointSubsetDataWithPC subsets
        """
        assert sum(split_ratio) == 1

        joints, gts, pcs = self.collect_all()


        subset = []
        split_ratio = [0.0] + split_ratio
        cumsum = np.cumsum(split_ratio)
        for prev_f, next_f in zip(cumsum[:-1], cumsum[1:]):
            prev_i, next_i = ceil(len(self) * prev_f), ceil(len(self) * next_f)
            subset.append(
                JointSubsetDataWithPC(
                    joints[prev_i:next_i],
                    gts[prev_i:next_i],
                    pcs[prev_i:next_i],
                    batchsize
                )
            )
        return subset

    def __len__(self):
        return sum([len(scene) for scene in self.scenes])


class DrDynamicSceneDataset(data.Dataset):
    """Dataset for a single scene with obstacle point cloud."""

    def __init__(self, scene_path, t="label", shuffle=True, num_points=100) -> None:
        """
        Initialize a single scene dataset.

        Args:
            scene_path: Path to the scene directory containing data.npz and obstacles.ply
            t: Type of data to load ('label' or 'distance')
            shuffle: Whether to shuffle the data
            num_points: Number of points to sample from obstacle point cloud
        """
        self.scene_path = pathlib.Path(scene_path)
        self.num_points = num_points
        self.t = t

        # Load collision/joint data
        data_file = self.scene_path / "data.npz"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        if t == "label":
            joints, gts = DrDynamicSceneDataset.load_label(data_file)
        elif t == "distance":
            joints, gts = DrDynamicSceneDataset.load_distance(data_file)
        else:
            raise NotImplementedError(t)

        # Load obstacle point cloud
        point_cloud = DrDynamicSceneDataset.load_pointcloud(
            self.scene_path / "obstacles.ply",
        )

        self.joints = torch.as_tensor(joints)
        self.gts = torch.as_tensor(gts)
        self.point_cloud = torch.as_tensor(point_cloud)

        if shuffle:
            reindex = torch.randperm(len(self.gts))
            self.joints = self.joints[reindex]
            self.gts = self.gts[reindex]

    @staticmethod
    def load_label(data_path):
        """Load collision labels from npz file."""
        data = np.load(data_path)
        joint = data.get("joint")
        collision = data.get("collision")
        return joint, collision[..., np.newaxis]

    @staticmethod
    def load_distance(data_path):
        """Load distance values from npz file."""
        data = np.load(data_path)
        joint = data.get("joint")
        distance = data.get("collision")
        return joint, distance

    @staticmethod
    def load_pointcloud(ply_path, num_points=-1):
        """
        Load point cloud from PLY file and randomly sample num_points.

        Args:
            ply_path: Path to the .ply file
            num_points: Number of points to sample

        Returns:
            Sampled point cloud array of shape (num_points, 3)
        """
        # Load PLY file
        plydata = PlyData.read(ply_path)

        # Extract vertex data
        vertex = plydata['vertex']
        points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

        if num_points > 0:
            # Randomly sample points
            if len(points) >= num_points > 0:
                indices = np.random.choice(len(points), num_points, replace=False)
            else:
                # If fewer points than needed, sample with replacement
                indices = np.random.choice(len(points), num_points, replace=True)
            
            sampled_points = points[indices]
        else:
            sampled_points = points
        return sampled_points

    def __getitem__(self, index: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample with its point cloud."""

        if isinstance(index, int):
            size = (self.num_points, )
        elif isinstance(index, slice):
            if index.start is None and index.stop is None: # all points
                size = (self.joints.size(0), self.num_points, )
            elif index.start is None:
                size = (index.stop, self.num_points, )
            elif index.stop is None:
                size = (self.joints.size(0) - index.start, self.num_points, )
            else:
                size = (index.stop - index.start, self.num_points, )

        # Replicate point cloud for all samples in this scene
        indices = torch.randint(self.point_cloud.size(0), size)
        self.point_cloud[indices]

        return (
            self.joints[index],
            self.gts[index] > 0,
            self.point_cloud[indices],
        )

    def __len__(self):
        return len(self.joints)


class JointSubsetDataWithPC(data.Dataset):
    """Subset of joint data with corresponding point clouds."""

    def __init__(
        self, joint: torch.Tensor, label: torch.Tensor,
        point_cloud: torch.Tensor, batch_size=100_000
    ) -> None:
        self.joints = joint.float().cuda().contiguous()
        self.labels = label.float().cuda().contiguous()
        self.point_clouds = point_cloud.float().cuda().contiguous()

        self.batch_size = batch_size
        self._current_idx = 0

    def shuffle(self):
        reindex = torch.randperm(len(self.labels))
        self.joints = self.joints[reindex]
        self.labels = self.labels[reindex]
        self.point_clouds = self.point_clouds[reindex]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index %= len(self)

        first_i, last_i = self.batch_size * index, min(
            self.batch_size * (index + 1), len(self.labels)
        )

        return (
            self.joints[first_i:last_i],
            self.labels[first_i:last_i],
            self.point_clouds[first_i:last_i],
        )

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

if __name__ == "__main__":
    ds = DrDynamicSceneDataset(
        "data/universal_robots_ur5e_dynamic_collision/sample_00000",
    )
    for pose, label, pc in ds:
        break

    ds = DrDynamicRobotData(
        "data/universal_robots_ur5e_dynamic_collision",
        "label",
        shuffle=True,
        num_points=100
    )
    
    train_ds, test_ds = ds.get_split()
    
    for i in train_ds:
        breakpoint()