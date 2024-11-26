import cv2
import torch
import numpy as np
import mujoco
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import copy
from typing import Dict, Tuple


class DummyCam:
    def __init__(
        self, azimuth: float, elevation: float, distance: float, lookat=[0, 0, 0]
    ):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = copy.deepcopy(lookat)  # Force lookat to be [0, 0, 0]


def generate_camera(dummy: DummyCam):
    # Configure the camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = dummy.distance
    cam.azimuth = dummy.azimuth
    cam.elevation = dummy.elevation
    cam.lookat = np.asarray(dummy.lookat)

    return cam


def put_pose_into_mujoco(model, data, pose):
    mujoco.mj_resetData(model, data)
    data.qpos = pose
    mujoco.mj_step(model, data)
    mujoco.mj_collision(model, data)


class ImageDemoDataset(Dataset):
    def __init__(self, reconstruction_dict: dict, background_color):

        self.reconstruction = reconstruction_dict

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.background_tuple = background_color

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        camera = self.reconstruction["camera"][index]
        image = self.reconstruction["image"][index]
        seg = self.reconstruction["segment"][index]
        bg = Image.new("RGB", image.size, self.background_tuple)
        depth = self.reconstruction["depth"][index] * np.asarray(seg)

        result = Image.composite(image, bg, seg)

        result = self.preprocess(result)
        seg = self.preprocess(seg)
        depth = self.preprocess(depth)

        return camera, result, seg, depth

    def __len__(self) -> int:
        return len(self.reconstruction["camera"])


def update_reconstruction_dict(renderer, data, dummy_cams, cams):
    reconstruction = {
        "camera": [],
        "image": [],
        "segment": [],
        "depth": [],
    }

    kernel = np.ones((3, 3), np.uint8)
    for dummy_cam, cam in zip(dummy_cams, cams):
        reconstruction["camera"].append(dummy_cam)

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        image = Image.fromarray(pixels)
        reconstruction["image"].append(image)

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        mask = (seg[:, :, 0] > 0) & (seg[:, :, 0] < 30)
        mask_eroded = cv2.erode(mask * 1.0, kernel, iterations=1)
        mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)
        image = Image.fromarray(mask_dilated.astype(bool))
        reconstruction["segment"].append(image)

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        depth[seg[:, :, 0] == -1] = 0
        reconstruction["depth"].append(depth)
    return reconstruction


def get_normalized_function(low, hight):
    lower_limits = torch.as_tensor(low, device="cuda")
    upper_limits = torch.as_tensor(hight, device="cuda")
    scale = 2 / (upper_limits - lower_limits)

    def normalized(joint_positions):
        return (joint_positions - lower_limits) * scale - 1.0

    def unnormalized(joint_positions):
        return (joint_positions + 1.0) / scale + lower_limits

    return normalized, unnormalized
