import os
os.environ['MUJOCO_GL'] = 'osmesa'

from collision.network import HyperNetwork
from collision.chain_utils import build_chain_relation_map
import torch
import numpy as np
import mujoco
import time
from PIL import Image
from tqdm import tqdm, trange

from pathlib import Path


class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def get_normalized_function(low, hight):
    lower_limits = np.asarray(low)
    upper_limits = np.asarray(hight)
    scale = 2 / (upper_limits - lower_limits)
    
    def normalized(joint_positions):
        return (joint_positions - lower_limits) * scale - 1.
    
    def unnormalized(joint_positions):
        return (joint_positions + 1.) / scale + lower_limits

    return normalized, unnormalized


def render_scene(model, data, sdf_model, cam, save_dir: Path, nums_joint=14):
    renderer = mujoco.Renderer(model, 480, 480)

    renderer.update_scene(data, camera=cam)

    # init scene: wait for object fall down
    mujoco.mj_step(model, data)

    def get_image(action):
        mujoco.mj_resetData(model, data)
        data.qpos = action
        mujoco.mj_step(model, data)
        mujoco.mj_collision(model, data)
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        return Image.fromarray(pixels)
    
    def sample_collision_pose():
        while True:
            pose = np.random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
            mujoco.mj_resetData(model, data)
            data.qpos = pose
            mujoco.mj_step(model, data)
            mujoco.mj_collision(model, data)
            if data.ncon == 1:
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render()
                return pose, Image.fromarray(pixels)
    
    iter_time = []
    consumes = []
    distance = []
    collision_before = []
    collision_after = []
    
    for iteration in trange(1000):
        collision_actions, image = sample_collision_pose()
        image.save("collision.png")
        collision_before.append(data.ncon)
    
        action_t = torch.nn.Parameter(
            torch.tensor(collision_actions, device='cuda', dtype=torch.float),
            requires_grad=True,
        )
        opt = torch.optim.Adam([action_t], lr=1)
        first = time.time()
        for i in range(1000):    
            loss, s = sdf_model(action_t[None])
            
            if loss < -0.000:
                break
            else:
                opt.zero_grad()
                loss.backward()
                opt.step()
        secend = time.time()
        consumes.append(secend-first)
        iter_time.append(i)

        not_collision_action = action_t.detach().cpu().numpy()
        distance.append(np.abs(not_collision_action-collision_actions).mean())

        image = get_image(not_collision_action)
        image.save("no collision.png")
        collision_after.append(data.ncon)

    iter_time = np.asarray(iter_time)
    consumes = np.asarray(consumes)
    distance = np.asarray(distance)
    collision_before = np.asarray(collision_before)
    collision_after = np.asarray(collision_after)

    print(f"""Iter {iter_time.mean()}
Time {consumes.mean()}
Dist {distance.mean()}
Before {collision_before.mean()}
After {collision_after.mean()}
Point {np.sum(collision_after>0)}""")

    np.savez('test.npz', time=consumes, iter_num=iter_time, distance=distance, collision_before=collision_before, collision_after=collision_after)

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

def generate_data(model_xml_dir: Path,
                  save_dir: Path, 
                  args):
    model_xml_path = model_xml_dir / "scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
    data = mujoco.MjData(model)

    relation_map, chain = build_chain_relation_map(model_xml_path.as_posix())
    sdf_model = HyperNetwork(chain.n_joints, relation_map)
    state_dict = torch.load('output/universal_robots_ur5e_collision/df_net.ckpt', weights_only=True)
    sdf_model.load_state_dict(state_dict)
    for parameters in sdf_model.parameters():
        parameters.requires_grad_(False)
    sdf_model.cuda()

    mujoco.mj_resetData(model, data)

    dummy_cam = DummyCam(0, -0, 2.5)
    cam = generate_camera(dummy_cam)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    render_scene(model, data, sdf_model, cam, save_dir)


if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_xml_dir', type=str, default="mujoco_menagerie/universal_robots_ur5e", help='Path to the model XML file.')
    args = parser.parse_args()

    model_xml_dir = Path(args.model_xml_dir)
    
    save_dir = Path("output") / "demonstration" / model_xml_dir.stem

    generate_data(model_xml_dir,
                  save_dir, 
                  args=args,)
