import os
os.environ['MUJOCO_GL'] = 'egl'

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../') #changing directories so that output/gsplat_full etc. exists

from pathlib import Path
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

from gaussian_renderer import render
import torch 
from PIL import Image
import numpy as np
import mujoco
from tqdm import tqdm, trange
from transformers import CLIPProcessor, CLIPModel
from IPython.display import display, clear_output
from torchvision.transforms import transforms
from itertools import cycle

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.to(device)
for param in clip_model.parameters():
    param.requires_grad = False

sys.argv = ['']
gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path='output/shadow_hand')

# load mujoco
model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'scene.xml'))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)


def sample_collision_pose():
    pose = np.zeros_like(model.jnt_range[:, 0])
    put_pose_into_mujoco(model, data, pose)
    return pose

# set camera
dummy_cams = [
    DummyCam(0, -90.0, 0.4, lookat=[0.15,  0.0, 0]),
    # DummyCam(0, -90.0, 0.4, lookat=[0.15,  0.1, 0]),
    # DummyCam(0, -90.0, 0.4, lookat=[0.15, -0.03, 0]),
    # DummyCam(0, -60.0, 0.4, lookat=[0.15,  0.0, 0]),
]
cams = [generate_camera(dummy_cam) for dummy_cam in dummy_cams]


renderer = mujoco.Renderer(model, 480, 480)
renderer.update_scene(data, camera=cams[0])

mujoco.mj_resetData(model, data)

pose = sample_collision_pose()

renderer.update_scene(data, camera=cams[0])
pixels = renderer.render()
image = Image.fromarray(pixels)
image

norm_fun, unnorm_fun = get_normalized_function(*kinematic_chain.get_joint_limits())

init_params = [0.0] * 24
joint_angles = torch.nn.Parameter(
    torch.tensor(init_params, dtype=torch.float32, device='cuda')
)
optimizer = torch.optim.Adam([joint_angles], lr=0.03)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

background_tuple = (255, 255, 255)
bg_color_t = torch.tensor(background_tuple).float().cuda() / 255.0

cam_list = []

for dummy_cam in dummy_cams:
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
    cam_list.append(
        Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 0.78, 0.78, 480, 480, joint_pose=norm_fun(joint_angles), zero_init=True).cuda()
    )

text_input = "A black robotic hand do OK gestures with white background"
with torch.no_grad():
    # text_input_t = clip.tokenize([text_input]).to(device)
    # embedding_input = clip_model.encode_text(text_input_t)
    inputs = clip_preprocess(text=[text_input], return_tensors="pt", padding=False)
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    embedding_input = clip_model.get_text_features(**inputs)

output_path = Path("output/shadow_hand")

relation_map, chain = build_chain_relation_map((output_path / "robot_xml/scene.xml").as_posix())
sdf_model = HyperNetwork(chain.n_joints, relation_map)
state_dict = torch.load(output_path / 'sdf_net.ckpt', weights_only=True)
sdf_model.load_state_dict(state_dict)
for parameters in sdf_model.parameters():
    parameters.requires_grad_(False)
sdf_model.cuda()
del state_dict

# Batch Exp
Exp_num = 10
sdf_thres = -0.1
collision_bf = []
collision_af = []
dot_product_bf = []
dot_product_af = []

topil = transforms.ToPILImage()

camera_pkg = cycle(cam_list)

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

max_iteration = 200


for i in trange(Exp_num):
    with torch.no_grad():
        joint_angles[:] = 0

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        camera = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render'].clamp(0, 1)
        depth_tensor = output_pkg['depth']

        image_embedding = clip_model.get_image_features(preprocess(image_tensor)[None])

        loss = -torch.matmul(image_embedding, embedding_input.T.detach())
        loss = loss / len(cam_list)

        loss.backward()

        if iteration % len(cam_list) == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        tbar.set_postfix({
            "loss": format(loss.item(), ".3f"),
        })

    put_pose_into_mujoco(model, data, joint_angles.detach().cpu().numpy())
    collision_bf.append(data.ncon)
    dot_product_bf.append(torch.matmul(image_embedding, embedding_input.T.detach()).item())

    tbar = trange(max_iteration, leave=False)
    for iteration in tbar:
        camera = next(camera_pkg)
        
        camera.joint_pose = norm_fun(joint_angles)

        output_pkg = render(camera, gaussians, bg_color_t)
        image_tensor = output_pkg['render'].clamp(0, 1)
        depth_tensor = output_pkg['depth']

        image_embedding = clip_model.get_image_features(preprocess(image_tensor)[None])

        sdf, s = sdf_model(joint_angles[None])

        loss = -torch.matmul(image_embedding, embedding_input.T.detach()) + torch.relu(sdf - sdf_thres) * 10
        
        loss = loss / len(cam_list) 

        loss.backward()

        if iteration % len(cam_list) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        tbar.set_postfix({
            "loss": format(loss.item(), ".3f"),
            "sdf": format(sdf.item(), ".3f"),
        })
    
    put_pose_into_mujoco(model, data, joint_angles.detach().cpu().numpy())
    collision_af.append(data.ncon)
    dot_product_af.append(torch.matmul(image_embedding, embedding_input.T.detach()).item())

collision_bf = np.array(collision_bf)
collision_af = np.array(collision_af)
dot_product_bf = np.array(dot_product_bf)
dot_product_af = np.array(dot_product_af)

print(f"""SDF {sdf_thres}
Collision Data
BF {collision_bf.mean()} {np.sum(collision_bf > 0)} 
AF {collision_af.mean()} {np.sum(collision_af > 0)}
Dot Product
BF {dot_product_bf.mean()} {dot_product_bf.std()}
AF {dot_product_af.mean()} {dot_product_af.std()}
""")