import os
os.environ['MUJOCO_GL'] = 'osmesa'


import numpy as np
import mujoco
from PIL import Image
from tqdm import tqdm

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


def render_scene(model, data, cam, save_dir: Path, nums_joint=6):
    renderer = mujoco.Renderer(model, 480, 480)

    renderer.update_scene(data, camera=cam)
    
    # data.qpos[:6] = np.ones(6)

    # init scene: wait for object fall down
    mujoco.mj_step(model, data, 1000)

    image_path = save_dir / "simulated"

    image_path.mkdir(parents=True, exist_ok=True)

    iteration = 0
    
    inverse_action = np.load("output/demonstration/universal_robots_ur5e/inverse_qpos.npy")

    norm_fun, unnorm_fun = get_normalized_function(model.jnt_range[:nums_joint, 0], model.jnt_range[:nums_joint, 1])

    for iteration, action in enumerate(tqdm(inverse_action)):
    
        data.ctrl = unnorm_fun(action)

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        
        image = Image.fromarray(pixels)
        image.save(image_path / f"{iteration:04d}.png")

        mujoco.mj_step(model, data, 1000)
        mujoco.mj_collision(model, data)

    os.system("ffmpeg -framerate 25 -i " + (image_path / r"%04d.png").as_posix() + " "   + (save_dir / "simulation.mp4").as_posix())

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
    args = parser.parse_args()

    model_xml_dir = Path(args.model_xml_dir)
    
    save_dir = Path("output") / "demonstration" / model_xml_dir.stem

    generate_data(model_xml_dir,
                  save_dir, 
                  args=args,)
