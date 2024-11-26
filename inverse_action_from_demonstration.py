import os


from collision.utils import get_normalized_function
from collision.chain_utils import build_chain_relation_map
from collision.network import SingleNetwork, HyperNetwork

from functools import wraps
from itertools import cycle
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

import json
from pathlib import Path


class DummyCam:
    def __init__(self, azimuth, elevation, distance, lookat=None):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0] if lookat is None else lookat

def optimize(gaussians, init_params, background_color, gt_pkg: dict, norm_fun, initial_lr=0.02, decay_factor=0.5, decay_steps=50, max_iteration=200, sdf_model=None):
    bg_color_t = torch.tensor(background_color).float().cuda() / 255.0

    camera_list, image_list, depth_list = gt_pkg["camera"], gt_pkg["image"], gt_pkg["depth"]

    image_list = [image.cuda() for image in image_list]
    depth_list = [depth.cuda() for depth in depth_list]

    joint_angles = torch.nn.Parameter(
        torch.tensor(init_params, dtype=torch.float32, requires_grad=True, device='cuda'),
          requires_grad=True)
    def get_gs_camera(camera_info: dict):
        dummy_cam = DummyCam(camera_info['azimuth'], camera_info['elevation'], camera_info['distance'], lookat=camera_info['lookat'])
        camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
        return Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=norm_fun(joint_angles), zero_init=True).cuda()
    
    camera_list = [get_gs_camera(camera_info) for camera_info in camera_list]
    optimizer = torch.optim.Adam([joint_angles], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    camera_pkg = cycle(zip(camera_list, image_list, depth_list))
    sdf = torch.zeros(1)

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        optimizer.zero_grad()

        camera, gt_image, gt_depth = next(camera_pkg)

        camera.joint_pose = joint_angles
        output_pkg = render(camera, gaussians, bg_color_t)
        gaussian_tensor = output_pkg['render']
        depth_tensor = output_pkg['depth']

        Ll2 = F.mse_loss(gaussian_tensor, gt_image)
        Ldepth = F.mse_loss(depth_tensor, gt_depth)

        loss = Ll2 + Ldepth
        loss = loss / len(camera_list)
        loss.backward()

        if iteration % len(camera_list):
            sdf, s = sdf_model(joint_angles[None])

            if sdf > 0:
                (0.1 * sdf).backward()

            optimizer.step()
            scheduler.step()

        tbar.set_postfix({
            "Iteration": format(iteration, "03d"),
            "L2": format(Ll2, ".5f"),
            "Ld": format(Ldepth, ".5f"),
            "LR": format(scheduler.get_last_lr()[0], ".5f"),
            "SDF": format(sdf.item(), ".5f"),
        })

    save_image(gaussian_tensor, 'test.png')

    return joint_angles.detach().cpu().numpy(), gaussian_tensor

@torch.no_grad()
def forward_render(gaussians, init_params, background_color):
    bg_color_t = torch.tensor(background_color).float().cuda() / 255.0

    joint_angles = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)

    azimuth, elevation, distance = 0, -10, 2.5 # Fixed camera parameters

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=joint_angles, zero_init=True).cuda()


    camera.joint_pose = joint_angles
    output_pkg = render(camera, gaussians, bg_color_t)
    gaussian_tensor = output_pkg['render']

    return gaussian_tensor

class ImageDemoDataset(Dataset):
    def __init__(self, data_path: Path, background_color):
        self.cameras = sorted(data_path.glob("[0-9][0-9]"))
        self.camera_data = {}
    
        for camera_path in self.cameras:
            with (camera_path / "camera.json").open('r') as f:
                camera_info = json.load(f)
            self.camera_data[camera_path] = {
                "camera_info": camera_info,
                "image": sorted(camera_path.glob("[0-9][0-9][0-9][0-9].png")),
                "mask": sorted(camera_path.glob("seg_[0-9][0-9][0-9][0-9].png")),
                "depth": sorted(camera_path.glob("depth_[0-9][0-9][0-9][0-9].npy")),
            }

        self.action = np.load(data_path / "qpos.npy")

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.background_tuple = background_color
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        camera_list = []
        image_list = []
        mask_list = []
        depth_list = []

        for camera_data in self.camera_data.values():
            camera_list.append(camera_data['camera_info'])

            image = Image.open(camera_data['image'][index])
            seg = Image.open(camera_data['mask'][index])
            bg = Image.new("RGB", image.size, self.background_tuple)
            depth = np.load(camera_data['depth'][index]) * np.asarray(seg)

            result = Image.composite(image, bg, seg)

            image_list.append(self.preprocess(result).float())
            mask_list.append(self.preprocess(seg).float())
            depth_list.append(torch.as_tensor(depth).float())
        
        return {
            "action": self.action[index],
            "camera": camera_list,
            "image": image_list,
            "mask": mask_list,
            "depth": depth_list,
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

    relation_map, chain = build_chain_relation_map((model_path / "robot_xml/scene.xml").as_posix())
    sdf_model = HyperNetwork(chain.n_joints, relation_map)
    state_dict = torch.load(model_path / 'sdf_net.ckpt', weights_only=True)
    sdf_model.load_state_dict(state_dict)
    for parameters in sdf_model.parameters():
        parameters.requires_grad_(False)
    sdf_model.cuda()
    del state_dict

    @generate_video(data_path, "forward")
    def forward():
        for iteration, data_pkg in enumerate(tqdm(demo_ds, desc="Forward Data Frame")):
            action = norm_fun(data_pkg["action"])
            render: torch.Tensor = forward_render(gaussians, action, background_tuple)
            save_image(render, data_path / "forward" / f"{iteration:04d}.png")
    
    @generate_video(data_path, "inverse")
    def backward():
        inverse_actions = []

        init_joint_angle = 14 * [0.0]

        for iteration, data_pkg in enumerate(tqdm(demo_ds, desc="Inverse Data Frame")):
            init_joint_angle, render_image = optimize(gaussians, 
                                                    init_joint_angle, 
                                                    background_tuple, 
                                                    data_pkg,
                                                    norm_fun,
                                                    initial_lr=0.005,
                                                    max_iteration=200,
                                                    sdf_model=sdf_model,
                                                    )
            inverse_actions.append(init_joint_angle)
            save_image(render_image, data_path / "inverse" / f"{iteration:04d}.png")
            
        np.save(data_path / "inverse_qpos.npy", np.stack(inverse_actions))
    
    # forward()
    backward()

if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', type=str, default="output/demonstration/universal_robots_ur5e_robotiq", help='Path to the demonstration file.')
    parser.add_argument('--model_path', type=str, default="output/universal_robots_ur5e_robotiq", help='Path to the demonstration file.')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    
    optimizer_pose(data_path,
                   model_path)



