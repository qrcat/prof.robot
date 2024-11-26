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

from torchvision.transforms import transforms


from pathlib import Path
from itertools import cycle

# load mujoco
model_xml_dir = Path("collision_scene/universal_robots_ur5e_scene2")
model_xml_path = model_xml_dir / "scene.xml"

model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

output_path = Path("output/universal_robots_ur5e_scene2_collision")

relation_map, chain = build_chain_relation_map(model_xml_path.as_posix())
sdf_model = HyperNetwork(chain.n_joints, relation_map)
state_dict = torch.load(output_path / 'sdf_net.ckpt', weights_only=True)
sdf_model.load_state_dict(state_dict)
for parameters in sdf_model.parameters():
    parameters.requires_grad_(False)
sdf_model.cuda()
del state_dict

def sample_collision_pose():
    pose = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
    put_pose_into_mujoco(model, data, pose)
    return pose

dummy_cams = [
    DummyCam(0, -45.0, 2.5),
    DummyCam(0, -60.0, 2.5),
    DummyCam(0, -75.0, 2.5),
    DummyCam(0, -90.0, 2.5),
]
cams = [generate_camera(dummy_cam) for dummy_cam in dummy_cams]

renderer = mujoco.Renderer(model, 480, 480)
renderer.update_scene(data, camera=cams[0])

sys.argv = ['']
gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path='output/universal_robots_ur5e_experiment')

background_tuple = (255, 255, 255)

norm_fun, unnorm_fun = get_normalized_function(*kinematic_chain.get_joint_limits())

init_params = [0.0] * 6
joint_angles = torch.nn.Parameter(
    torch.tensor(init_params, dtype=torch.float32, device='cuda')
)
optimizer = torch.optim.Adam([joint_angles], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

bg_color_t = torch.tensor(background_tuple).float().cuda() / 255.0

topil = transforms.ToPILImage()

# Batch Exp
Exp_num = 100
sdf_thres = -0.0
collision_data = []
collision_bf = []
collision_af = []


for i in trange(Exp_num):
    while True:
        pose = sample_collision_pose()
        if data.ncon > 0:
            break
    collision_data.append(data.ncon)
    continue

    with torch.no_grad():
        joint_angles[:] = torch.tensor(pose)
        joint_angles += 0.1 * torch.rand_like(joint_angles)

    background_tuple = (255, 255, 255)
    reconstruction = update_reconstruction_dict(renderer, data, dummy_cams, cams)
    dataset = ImageDemoDataset(reconstruction, background_color=(255, 255, 255))

    bg_color_t = torch.tensor(background_tuple).float().cuda() / 255.0

    cam_list = []
    mask_list = []
    image_list = []
    depth_list = []

    for dummy_cam, image, segment, depth in dataset:
        camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
        cam_list.append(
            Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=norm_fun(joint_angles), zero_init=True).cuda()
        )
        mask_list.append(segment[0].bool().cuda())
        image_list.append(image.cuda())
        depth_list.append(depth[0].cuda())
    
    max_iteration = 100

    camera_pkg = cycle(zip(cam_list, mask_list, image_list, depth_list))

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        camera, gt_mask, gt_image, gt_depth = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render']
        depth_tensor = output_pkg['depth']

        Ll2 = F.mse_loss(image_tensor, gt_image)
        Ldepth = F.mse_loss(depth_tensor, gt_depth)
        Ll2_s = F.mse_loss(image_tensor[:, gt_mask], gt_image[:, gt_mask])
        Ldepth_s = F.mse_loss(depth_tensor[gt_mask], gt_depth[gt_mask])


        loss = 0.1 * Ll2 + 0.1 * Ldepth + Ll2_s + Ldepth_s
        loss = loss / len(cam_list)

        loss.backward()

        if iteration % len(cam_list) == 0:
            optimizer.step()
            optimizer.zero_grad()

        tbar.set_postfix({
            "L2": format(Ll2, ".3f"),
            "Ld": format(Ldepth, ".3f"),
            "LR": format(scheduler.get_last_lr()[0], ".3f"),
        })

    put_pose_into_mujoco(model, data, joint_angles.detach().cpu().numpy())
    collision_bf.append(data.ncon)

    max_iteration = 200

    topil = transforms.ToPILImage()

    camera_pkg = cycle(zip(cam_list, mask_list, image_list, depth_list))

    tbar = trange(max_iteration, leave=False)
    iteration = 0
    optimizer.zero_grad()
    while iteration < 10000:
        camera, gt_mask, gt_image, gt_depth = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render']
        depth_tensor = output_pkg['depth']

        Ll2 = F.mse_loss(image_tensor, gt_image)
        Ldepth = F.mse_loss(depth_tensor, gt_depth)
        Ll2_s = F.mse_loss(image_tensor[:, gt_mask], gt_image[:, gt_mask])
        Ldepth_s = F.mse_loss(depth_tensor[gt_mask], gt_depth[gt_mask])

        sdf, s = sdf_model(joint_angles[None])
        loss = 0.1 * Ll2 + 0.1 * Ldepth + Ll2_s + Ldepth_s + sdf * iteration / 5

        if sdf < sdf_thres:
            break

        loss = loss / len(cam_list)
        
        loss.backward()

        if iteration % len(cam_list) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        tbar.set_postfix({
            "SDF": format(sdf.item(), ".3f"),
            "L2": format(Ll2, ".3f"),
            "Ld": format(Ldepth, ".3f"),
            "LR": format(scheduler.get_last_lr()[0], ".3f"),
        })
        iteration += 1
        tbar.update()
    
    put_pose_into_mujoco(model, data, joint_angles.detach().cpu().numpy())
    collision_af.append(data.ncon)

    tqdm.write(f"After: {sum(collision_af)}/{len(collision_af)} Before: {sum(collision_bf)}/{len(collision_bf)}")

collision_data = np.asarray(collision_data)
collision_bf = np.asarray(collision_bf)
collision_af = np.asarray(collision_af)

print(f"""Data Nums {Exp_num} SDF {sdf_thres} Collision {collision_data.mean():.6f}
Before:
    Point\tRate
    {collision_bf.mean():.6f}\t{np.sum(collision_bf>0)/len(collision_bf):.6f}
After:
    Point\tRate
    {collision_af.mean():.6f}\t{np.sum(collision_af>0)/len(collision_af):.6f}
""")

