import os
os.environ['MUJOCO_GL'] = 'egl'

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../') #changing directories so that output/gsplat_full etc. exists

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from collision.utils import DummyCam, ImageDemoDataset, generate_camera, put_pose_into_mujoco, update_reconstruction_dict, get_normalized_function
from utils.mujoco_utils import compute_camera_extrinsic_matrix
from scene.cameras import Camera_Pose
from collision.chain_utils import build_chain_relation_map
from collision.network import SingleNetwork, HyperNetwork
from contextlib import redirect_stdout
from video_api import initialize_gaussians

import cv2
from gaussian_renderer import render
import sys 
import torch 
from PIL import Image
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm, trange
from mujoco.usd import exporter

from IPython.display import display, clear_output
from torchvision.transforms import transforms


from pathlib import Path
from itertools import cycle


# load mujoco
model_xml_dir = Path("collision_scene/universal_robots_ur5e_scene2")
model_xml_path = model_xml_dir / "scene.xml"

model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)


def sample_collision_pose():
    pose = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
    put_pose_into_mujoco(model, data, pose)
    return pose

# set camera
dummy_cams = [
    DummyCam(0, -45.0, 2.5),
    DummyCam(0, -60.0, 2.5),
    DummyCam(0, -75.0, 2.5),
    DummyCam(0, -90.0, 2.5),
]
cams = [generate_camera(dummy_cam) for dummy_cam in dummy_cams]

renderer = mujoco.Renderer(model, 480, 480)
renderer.update_scene(data, camera=cams[0])
background_tuple = (255, 255, 255)


output_path = Path("output/universal_robots_ur5e_scene2_collision")

relation_map, chain = build_chain_relation_map(model_xml_path.as_posix())
sdf_model = HyperNetwork(chain.n_joints, relation_map)
state_dict = torch.load(output_path / 'sdf_net.ckpt', weights_only=True)
sdf_model.load_state_dict(state_dict)
for parameters in sdf_model.parameters():
    parameters.requires_grad_(False)
sdf_model.cuda()
del state_dict

sys.argv = ['']
gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path='output/universal_robots_ur5e_experiment')

norm_fun, unnorm_fun = get_normalized_function(*kinematic_chain.get_joint_limits())



bg_color_t = torch.tensor(background_tuple).float().cuda() / 255.0


# Batch Exp
Exp_num = 10
sdf_thres = -0.1
collision_bf = []
collision_af = []

for i in trange(Exp_num):
    while True:
        init_pose = sample_collision_pose()
        if data.ncon == 0:
            break

    while True:
        final_pose = sample_collision_pose()
        if data.ncon == 0:
            break

    init_pose_t = torch.tensor(init_pose[None], dtype=torch.float32, device='cuda')
    finl_pose_t = torch.tensor(final_pose[None], dtype=torch.float32, device='cuda')

    reconstruction = update_reconstruction_dict(renderer, data, dummy_cams, cams)
    finl_dataset = ImageDemoDataset(reconstruction, background_color=(255, 255, 255))

    cam_list = []
    mask_list = []
    image_list = []
    depth_list = []

    for dummy_cam, image, segment, depth in finl_dataset:
        camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
        cam_list.append(
            Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=norm_fun(init_pose_t), zero_init=True).cuda()
        )
        mask_list.append(segment[0].bool().cuda())
        image_list.append(image.cuda())
        depth_list.append(depth[0].cuda())
    camera_pkg = cycle(zip(cam_list, mask_list, image_list, depth_list))

    joint_angles = torch.nn.Parameter(init_pose_t[0].clone().detach().requires_grad_(True))
    optimizer = torch.optim.Adam([joint_angles], lr=0.05)

    max_iteration = 400

    intp_pose = []

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        camera, gt_mask, gt_image, gt_depth = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render']
        depth_tensor = output_pkg['depth']

        Ll2_s = F.mse_loss(image_tensor[:, gt_mask], gt_image[:, gt_mask])
        Ldepth_s = F.mse_loss(depth_tensor[gt_mask], gt_depth[gt_mask])

        sdf, s = sdf_model(joint_angles[None])

        pose_loss = torch.nn.functional.l1_loss(joint_angles, finl_pose_t)

        loss = Ll2_s + Ldepth_s + pose_loss * 10
        loss = loss / len(cam_list)

        loss.backward()

        if iteration % len(cam_list) == 0:
            optimizer.step()
            optimizer.zero_grad()
            intp_pose.append(joint_angles.detach().cpu().numpy())
        
        tbar.set_postfix({
            "L2": format(Ll2_s, ".3f"),
            "Ld": format(Ldepth_s, ".3f"),
            "SDF": format(sdf.item(), ".3f"),
            "Angle": format(pose_loss.item(), ".3f"),
        })

    for action in intp_pose:
        put_pose_into_mujoco(model, data, action)
        collision_bf.append(data.ncon) 

    with torch.no_grad():
        joint_angles[:] = init_pose_t[0]
    
    intp_pose = []
    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        camera, gt_mask, gt_image, gt_depth = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render']
        depth_tensor = output_pkg['depth']

        Ll2_s = F.mse_loss(image_tensor[:, gt_mask], gt_image[:, gt_mask])
        Ldepth_s = F.mse_loss(depth_tensor[gt_mask], gt_depth[gt_mask])

        sdf, s = sdf_model(joint_angles[None])

        pose_loss = (joint_angles - finl_pose_t).abs().mean()

        loss = Ll2_s + Ldepth_s + torch.relu(sdf-sdf_thres) * 5 + pose_loss * 10
        loss = loss / len(cam_list)

        loss.backward()

        if iteration % len(cam_list) == 0:

            optimizer.step()
            optimizer.zero_grad()
            intp_pose.append(joint_angles.detach().cpu().numpy())
        
        tbar.set_postfix({
            "L2": format(Ll2_s, ".3f"),
            "Ld": format(Ldepth_s, ".3f"),
            "SDF": format(sdf.item(), ".3f"),
            "Angle": format(pose_loss.item(), ".3f"),
        })

    target_pose = torch.nn.Parameter(
        torch.tensor(np.asarray(intp_pose), dtype=torch.float32, device='cuda')
    )
    optimizer = torch.optim.Adam([target_pose], lr=0.1)

    tbar = trange(1000, leave=True)
    for i in tbar:
        sdf_loss = torch.relu(sdf_model(target_pose)[0]-sdf_thres).mean()
        tv_loss = (torch.cat([init_pose_t, target_pose[:-1]], dim=0) - torch.cat([target_pose[:-1], finl_pose_t], dim=0)).abs().mean()

        loss = sdf_loss + tv_loss

        tbar.set_postfix({
            "SDF": format(sdf_loss.item(), ".5f"),
            "TV": format(tv_loss.item(), ".5f"),
        })

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for action in target_pose.detach().cpu().numpy():
        put_pose_into_mujoco(model, data, joint_angles.detach().cpu().numpy())
        collision_af.append(data.ncon)

collision_bf = np.array(collision_bf)
collision_af = np.array(collision_af)

print(f"""SDF Level {sdf_thres}
Collision Data
BF {collision_bf.mean()} {np.sum(collision_bf > 0)} 
AF {collision_af.mean()} {np.sum(collision_af > 0)}
""")