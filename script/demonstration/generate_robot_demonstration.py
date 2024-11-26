import os
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import mujoco
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import copy
import json
from pathlib import Path


class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def render_scene(model, data, save_dir: Path, actions, *cam_list, nums_joint=7):
    def render(cam: mujoco.MjvCamera, path: Path):
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        
        image = Image.fromarray(pixels)
        image.save(path / f"{iteration:04d}.png")

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        image = Image.fromarray((seg[:, :, 0]>0)&(seg[:, :, 0]<58))
        image.save(path / f"seg_{iteration:04d}.png")

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        depth[seg[:, :, 0]==-1] = 0
        np.save(path / f"depth_{iteration:04d}.npy", depth)

        plt.figure(figsize=depth.shape[::-1], dpi=1)
        plt.imshow(depth, cmap='viridis')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.savefig(path / f"depth_{iteration:04d}.jpg")
        plt.close()

    renderer = mujoco.Renderer(model, 480, 480)

    # init scene: wait for object fall down
    mujoco.mj_step(model, data, 1000)

    for cam_id, cam in enumerate(cam_list):
        cam_dir = save_dir / f"{cam_id:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        with (cam_dir / "camera.json").open('w') as f:
            json.dump({
                "azimuth": cam.azimuth,
                "elevation": cam.elevation,
                "distance": cam.distance,
                "lookat": cam.lookat.tolist(),
            }, f)
        

    qpos = []
    
    iteration = 0
    for action in tqdm(actions):
        action, iters = action[:-1], int(action[-1])
        data.ctrl = action
        for _ in trange(iters):
            
            for cam_id, cam in enumerate(cam_list):
                render(cam, save_dir / f"{cam_id:02d}")

            qpos.append(copy.deepcopy(data.qpos))

            mujoco.mj_step(model, data, 10)
            mujoco.mj_collision(model, data)

            iteration += 1

    for cam_id, _ in enumerate(cam_list):
        os.system("ffmpeg -framerate 25 -i " + (save_dir / f"{cam_id:02d}" / r"%04d.png").as_posix() + " "  + (save_dir / f"demonstration_{cam_id:02d}.mp4").as_posix() + " -y")

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

def get_action(path: Path):
    actions = []
    with path.open() as f:
        while l := f.readline():
            try:
                action = eval(l.strip())
                action = np.asarray(action)
            except:
                continue

            actions.append(action)

    return actions

def generate_data(model_xml_dir: Path,
                  save_dir: Path, 
                  args):
    model_xml_path = model_xml_dir / "scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    dummy_cams = [
        DummyCam(0,  -0.0, 2.50),
        DummyCam(0, -10.0, 1.00),
        # DummyCam(0, -30.0, 1.50),
        # DummyCam(0, -45.0, 1.00),
        # DummyCam(0, -60.0, 1.50),
        # DummyCam(0, -80.0, 2.00),
        DummyCam(0, -90.0, 2.50),
    ]
    cams = [generate_camera(dummy_cam) for dummy_cam in dummy_cams]

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    actions = get_action(model_xml_dir / "action_list.txt")

    render_scene(model, data, save_dir, actions, *cams)

if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_xml_dir', type=str, default="mujoco_demo_control/universal_robots_ur5e_robotiq", help='Path to the model XML file.')
    args = parser.parse_args()

    model_xml_dir = Path(args.model_xml_dir)
    
    save_dir = Path("output") / "demonstration" / model_xml_dir.stem

    generate_data(model_xml_dir,
                  save_dir, 
                  args=args,)
