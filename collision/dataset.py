import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm, trange

import pathlib
from math import ceil
from typing import Any, List, Tuple
from multiprocessing.pool import ThreadPool


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


def get_qpos_and_ctrl(path: pathlib.Path, tbar):
    data = np.load(path)

    qpos = data["qpos"]
    ctrl = data["ctrl"]

    tbar.update(1)

    return qpos, ctrl


def get_data(path: pathlib.Path, tbar):
    collision_l = []
    collision_n = []

    data = np.load(path)

    collision_l = data["joint"]
    collision_n = data["collision"]

    tbar.update(1)

    return collision_l, collision_n


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/universal_robots_ur5e_collision"
    )
    args = parser.parse_args(
        ["--data_path", "data/universal_robots_ur5e_scene3_collision"]
    )
    ##################################################################################
    data_path = pathlib.Path(args.data_path)
    print("load pkl data from", data_path)

    npz_files = list(data_path.glob("sample_*/data.npz"))
    tbar = tqdm(npz_files, "Read NPZ")

    with ThreadPool() as pool:
        OUT = pool.map(lambda x: (get_data(x, tbar)), npz_files)
        pool.close()
        pool.join()

    collision_j = np.concatenate([x[0] for x in OUT], 0)
    collision_n = np.concatenate([x[1] for x in OUT], 0)
    ##################################################################################
    joint_data_path = data_path / "joint_data.npz"
    print("save joint data", joint_data_path)
    np.savez(joint_data_path, joint=collision_j, nums=collision_n)
    ##################################################################################
    with torch.no_grad():
        collision_j_t = torch.tensor(collision_j).float()  # N, 19
        collision_n_t = torch.tensor(collision_n)  # N,

        new_quat_wo_c = collision_j_t[collision_n_t < 1].cuda().contiguous()
        new_quat_w__c = collision_j_t[collision_n_t > 0].cuda().contiguous()
        print("F/T", new_quat_w__c.shape[0], new_quat_wo_c.shape[0])
    ##################################################################################
    import faiss
    import faiss.contrib.torch_utils

    print("use knn to compute distance by faiss, it takes lots of time...")
    res = faiss.StandardGpuResources()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        D, I = faiss.knn_gpu(res, new_quat_w__c, new_quat_wo_c, 1, use_raft=False)
    print("compute successful")

    dist_w__c = D.cpu().detach().numpy() / collision_j.shape[1]
    dist_wo_c = np.zeros((len(new_quat_wo_c), 1), dtype=np.float32)

    joint = np.concatenate(
        [new_quat_w__c.cpu().detach().numpy(), new_quat_wo_c.cpu().detach().numpy()], 0
    )
    dist = np.concatenate([dist_w__c, dist_wo_c], 0)

    dist_data_path = data_path / "dist_data.npz"
    print("save distance data", dist_data_path)
    np.savez(dist_data_path, joint=joint, dist=dist)
    ##################################################################################
    import matplotlib.pyplot as plt

    plt.hist(dist, bins=100)
    plt.savefig(data_path / "dist.png")
    plt.close()
