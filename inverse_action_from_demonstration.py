from functools import wraps
import os
from typing import Dict, Tuple

import numpy as np
from utils.mujoco_utils import compute_camera_extrinsic_matrix
from tqdm import tqdm, trange

from video_api import initialize_gaussians
from gaussian_renderer import render
from scene.cameras import Camera_Pose
import torch
import torch.nn.functional as F

import os
import numpy as np
import mujoco
from PIL import Image

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.transforms import transforms

from pathlib import Path


class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def generate_camera(dummy: DummyCam):
    # Configure the camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = dummy.distance
    cam.azimuth = dummy.azimuth
    cam.elevation = dummy.elevation
    cam.lookat = np.asarray(dummy.lookat)

    return cam

def optimize(gaussians, init_params, background_color, gt_pkg: dict, initial_lr=0.02, decay_factor=0.95, decay_steps=50, max_iteration=200):
    bg_color_t = torch.tensor(background_color).float().cuda() / 255.0

    gt_image = gt_pkg['image'].cuda()
    gt_mask = gt_pkg['mask'][..., None].cuda()

    joint_angles = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)

    azimuth, elevation, distance = 0, -45, 3  # Fixed camera parameters

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=joint_angles, zero_init=True).cuda()

    optimizer = torch.optim.Adam([joint_angles], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        optimizer.zero_grad()

        camera.joint_pose = joint_angles
        output_pkg = render(camera, gaussians, bg_color_t)
        gaussian_tensor = output_pkg['render']
        alpha_tensor = output_pkg['alpha']

        Ll2 = F.mse_loss(gaussian_tensor, gt_image)
        Lalpha = F.cross_entropy(alpha_tensor, gt_mask)

        loss = Ll2 + 0.01*Lalpha
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        tbar.set_postfix({
            "Iteration": format(iteration, "03d"),
            "L2": format(Ll2, ".5f"),
            "La": format(Lalpha, ".5f"),
            "LR": format(scheduler.get_last_lr()[0], ".5f"),
        })

    save_image(gaussian_tensor, 'test.png')

    return joint_angles.detach().cpu().numpy(), gaussian_tensor

def get_normalized_function(low, hight):
    lower_limits = np.asarray(low)
    upper_limits = np.asarray(hight)
    scale = 2 / (upper_limits - lower_limits)
    
    def normalized(joint_positions):
        return (joint_positions - lower_limits) * scale - 1.
    
    def unnormalized(joint_positions):
        return (joint_positions + 1.) / scale + lower_limits

    return normalized, unnormalized

@torch.no_grad()
def forward_render(gaussians, init_params, background_color):
    bg_color_t = torch.tensor(background_color).float().cuda() / 255.0

    joint_angles = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)

    azimuth, elevation, distance = 0, -45, 3  # Fixed camera parameters

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=joint_angles, zero_init=True).cuda()


    camera.joint_pose = joint_angles
    output_pkg = render(camera, gaussians, bg_color_t)
    gaussian_tensor = output_pkg['render']

    return gaussian_tensor

class ImageDemoDataset(Dataset):
    def __init__(self, data_path: Path, background_color):
        self.images_path = sorted(data_path.glob("image/*.png"))
        self.segments_path = sorted(Path(data_path).glob("seg/*.png"))

        self.action = np.load(data_path / "qpos.npy")

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.background_tuple = background_color
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        image = Image.open(self.images_path[index])
        seg = Image.open(self.segments_path[index])
        bg = Image.new("RGB", image.size, self.background_tuple)

        result = Image.composite(image, bg, seg)
        
        return {
            "action": self.action[index],
            "image": self.preprocess(result).float(),
            "mask": self.preprocess(seg).float(),
        }
    
    def __len__(self) -> int:
        return len(self.images_path)


def generate_video(path: Path, name: str):
    def decorator(func):
        @wraps(func)
        def wrapFunc():
            data_path = path / name
            data_path.mkdir(parents=True, exist_ok=True)
            func()
            os.system("ffmpeg -framerate 25 -i " + (data_path / r"%04d.png").as_posix() + " "  + (path / f"{name}.mp4").as_posix() + " -y")
        return wrapFunc
    return decorator


def optimizer_pose(data_path: Path, model_path: Path):
    gaussians, _, sample_cameras, kinematic_chain = initialize_gaussians(model_path)

    background_tuple = (255, 255, 255)

    demo_ds = ImageDemoDataset(data_path, background_tuple)

    norm_fun, unnorm_fun = get_normalized_function(*kinematic_chain.get_joint_limits())

    @generate_video(data_path, "forward")
    def forward():
        for iteration, data_pkg in enumerate(tqdm(demo_ds, desc="Forward Data Frame")):
            action = norm_fun(data_pkg["action"])
            render: torch.Tensor = forward_render(gaussians, action, background_tuple)
            save_image(render, data_path / "forward" / f"{iteration:04d}.png")
    
    @generate_video(data_path, "inverse")
    def backward():
        inverse_actions = []

        init_joint_angle = 6 * [0.0]

        for iteration, data_pkg in enumerate(tqdm(demo_ds, desc="Inverse Data Frame")):
            init_joint_angle, render_image = optimize(gaussians, 
                                                    init_joint_angle, 
                                                    background_tuple, 
                                                    data_pkg,
                                                    initial_lr=0.01)
            inverse_actions.append(init_joint_angle)
            save_image(render_image, data_path / "inverse" / f"{iteration:04d}.png")
            
        np.save(data_path / "inverse_qpos.npy", np.stack(inverse_actions))
    
    forward()
    backward()

if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', type=str, default="output/demonstration/universal_robots_ur5e", help='Path to the demonstration file.')
    parser.add_argument('--model_path', type=str, default="output/universal_robots_ur5e_experiment", help='Path to the demonstration file.')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    
    optimizer_pose(data_path,
                   model_path)



