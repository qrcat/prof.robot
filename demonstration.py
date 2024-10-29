"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_realtime.py --model_path output/[path_to_your_model_directory]
"""


import sys
import tempfile

import os
os.environ['MUJOCO_GL'] = 'egl'
import queue
import threading

from utils.mujoco_utils import compute_camera_extrinsic_matrix, extract_camera_parameters

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

import numpy as np
import mujoco
from utils.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from tqdm import tqdm

from video_api import initialize_gaussians
from gaussian_renderer import render
from scene.cameras import Camera_Pose, Camera
import torch
import torch.nn.functional as F
import time

import os
import tempfile
import gradio as gr
import numpy as np
import mujoco
from utils.mujoco_utils import simulate_mujoco_scene
from PIL import Image



from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'


class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*joint_angles):
    azimuth, elevation, distance = 0, -45, 3  # Fixed camera parameters
    
    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    joint_angles = torch.tensor(joint_angles)
    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), example_camera.FoVx, example_camera.FoVy,\
                            480, 480, joint_pose=joint_angles, zero_init=True).cuda()
    frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def reset_params():
    new_input_received.set()
    return [0.0] * n_joints

def initial_render():
    initial_params = reset_params()
    print("initial_params: ", initial_params, "rendering scene...")
    mujoco_image = render_scene(*initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    print("done rendering scene")

    optimization_queue.put((initial_params, initial_params))

    return mujoco_image, gaussian_image

def render_scene(model, data, cam, save_dir: Path, nums_joint=6):
    renderer = mujoco.Renderer(model, 480, 480)

    renderer.update_scene(data, camera=cam)
    
    # data.qpos[:6] = np.ones(6)

    # init scene: wait for object fall down
    mujoco.mj_step(model, data, 1000)

    image_path = save_dir / "image"
    seg_path = save_dir / "seg"

    image_path.mkdir(parents=True, exist_ok=True)
    seg_path.mkdir(parents=True, exist_ok=True)

    qpos = []
    
    iteration = 0
    while True:
        if iteration == 0:
            data.ctrl = np.array([-1.82, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif iteration == 40:
            data.ctrl = np.array([-1.82, 0.0, 2.0, 0.0, 0.0, 0.0])
        if iteration > 100:
            break


        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        
        image = Image.fromarray(pixels)
        image.save(image_path / f"{iteration:04d}.png")

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)

        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        image = Image.fromarray((seg[:, :, 0]>0)&(seg[:, :, 0]<30))
        image.save(seg_path / f"{iteration:04d}.png")

        qpos.append(data.qpos[:nums_joint])

        mujoco.mj_step(model, data, 10)
        mujoco.mj_collision(model, data)

        iteration += 1

    qpos = np.stack(qpos)
    np.save(save_dir / "qpos.npy", qpos)
    
    del renderer

def generate_camera(dummy: DummyCam):
    # Configure the camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = dummy.distance
    cam.azimuth = dummy.azimuth
    cam.elevation = dummy.elevation
    cam.lookat = np.asarray(dummy.lookat)

    return cam

def generate_data(num_actors, 
                  model_xml_dir: Path,
                  save_dir: Path, 
                  args):
    model_xml_path = model_xml_dir / "scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    dummy_cam = DummyCam(0, -45, 3.0)
    cam = generate_camera(dummy_cam)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    render_scene(model, data, cam, save_dir)

if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_xml_dir', type=str, default="demonstration/universal_robots_ur5e", help='Path to the model XML file.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples.')
    parser.add_argument('--num_actors', type=int, default=20, help='Number of actors.')
    args = parser.parse_args()

    model_xml_dir = Path(args.model_xml_dir)
    
    num_actors = args.num_actors
    num_samples = args.num_samples
    
    save_dir = Path("output") / "demonstration" / model_xml_dir.stem

    generate_data(num_actors, 
                  model_xml_dir,
                  save_dir, 
                  args=args,)



