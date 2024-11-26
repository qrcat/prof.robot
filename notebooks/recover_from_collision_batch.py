import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
import time
import torch 
from PIL import Image
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import CLIPProcessor, CLIPModel
from IPython.display import display, clear_output
from torchvision.transforms import transforms


from pathlib import Path
from itertools import cycle

# load mujoco
model_xml_path = Path("collision_scene/universal_robots_ur5e_scene2/scene.xml")
model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)


def sample_collision_pose():
    pose = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
    put_pose_into_mujoco(model, data, pose)
    return pose

# set camera
dummy_cams = [
    DummyCam(0, -45.0, 2.5, lookat=[0,  0, 0]),
]
cams = [generate_camera(dummy_cam) for dummy_cam in dummy_cams]

renderer = mujoco.Renderer(model, 480, 480)
renderer.update_scene(data, camera=cams[0])

mujoco.mj_resetData(model, data)

relation_map, chain = build_chain_relation_map(model_xml_path.as_posix())
sdf_model = HyperNetwork(chain.n_joints, relation_map)
state_dict = torch.load('output/universal_robots_ur5e_robotiq/sdf_net.ckpt', weights_only=True)
sdf_model.load_state_dict(state_dict)
for parameters in sdf_model.parameters():
    parameters.requires_grad_(False)
sdf_model.cuda()
del state_dict

sdf_we_model = HyperNetwork(chain.n_joints, relation_map)
state_dict = torch.load('output/universal_robots_ur5e_robotiq/sdf_net_wo_eik.ckpt', weights_only=True)
sdf_we_model.load_state_dict(state_dict)
for parameters in sdf_model.parameters():
    parameters.requires_grad_(False)
sdf_we_model.cuda()
del state_dict

def get_p(sdf_m, joint_angles):
    sdf, s = sdf_m(joint_angles[None])
    return torch.sigmoid(sdf * s)

# Batch Exp
Exp_num = 100
sdf_thres = -0.1
iter_time_no_e = []
iter_time_ours = []
consumes_no_e = []
consumes_ours = []
distance_no_e = []
distance_ours = []
collision_no_e = []
collision_ours = []
gradient_ours = []
gradient_no_e = []

MAX_ITER = 1000

for i in trange(Exp_num):
    while True:
        pose = sample_collision_pose()
        if data.ncon > 0:
            break
        


    joint_angles = torch.tensor(pose, dtype=torch.float32).cuda()
    action_t = torch.nn.Parameter(joint_angles, requires_grad=True,)
    optimize = torch.optim.Adam([action_t], lr=0.01)
    first = time.time()
    for i in range(MAX_ITER):    
        sdf, s = sdf_model(action_t[None])
        
        if sdf < -0.100:
            break
        else:
            optimize.zero_grad()
            sdf.backward()
            optimize.step()
            
            gradient_ours.append(action_t.grad.norm().detach().cpu().numpy())
    
    secend = time.time()

    consumes_ours.append(secend-first)
    iter_time_ours.append(i)
    distance_ours.append(np.abs(action_t.detach().cpu().numpy()-pose).mean())

    data.qpos = action_t.detach().cpu().numpy()
    mujoco.mj_step(model, data)
    mujoco.mj_collision(model, data)
    collision_ours.append(data.ncon)

    # # Without Eikonal
    # joint_angles = torch.tensor(pose, dtype=torch.float32).cuda()
    # action_t = torch.nn.Parameter(joint_angles, requires_grad=True,)
    # optimize = torch.optim.Adam([action_t], lr=10)
    # first = time.time()
    # for i in range(MAX_ITER):    
    #     p = get_p(sdf_we_model, action_t)

    #     if p < 0.5:
    #         break
    #     else:
    #         optimize.zero_grad()
    #         p.backward()
    #         optimize.step()
    #         gradient_no_e.append(action_t.grad.norm().detach().cpu().numpy())

    # secend = time.time()

    # consumes_no_e.append(secend-first)
    # iter_time_no_e.append(i)
    # distance_no_e.append(np.abs(action_t.detach().cpu().numpy()-pose).mean())

    # data.qpos = action_t.detach().cpu().numpy()
    # mujoco.mj_step(model, data)
    # mujoco.mj_collision(model, data)
    # collision_no_e.append(data.ncon)

collision_no_e = np.array(collision_no_e)
collision_ours = np.array(collision_ours)

print(f"""
without eikonal:
    iter: {np.mean(iter_time_no_e)}
    time: {np.mean(consumes_no_e)}
    distance: {np.mean(distance_no_e)}
    Collision Point: {np.mean(collision_no_e)}
    Collision Rate: {np.mean(collision_no_e>0)}
    gradient: {np.mean(gradient_no_e)}

with eikonal:
    iter: {np.mean(iter_time_ours)}
    time: {np.mean(consumes_ours)}
    distance: {np.mean(distance_ours)}
    Collision Point: {np.mean(collision_ours)}
    Collision Rate: {np.mean(collision_ours>0)}
    gradient: {np.mean(gradient_ours)}
""")