import os
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import mujoco
from PIL import Image

import copy
from pathlib import Path



class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

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

        qpos.append(copy.deepcopy(data.qpos[:nums_joint]))

        mujoco.mj_step(model, data, 10)
        mujoco.mj_collision(model, data)

        iteration += 1

    os.system("ffmpeg -framerate 25 -i " + (image_path / r"%04d.png").as_posix() + " "  + (save_dir / "demonstration.mp4").as_posix())

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
    import argparse 

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_xml_dir', type=str, default="demonstration/universal_robots_ur5e", help='Path to the model XML file.')
    args = parser.parse_args()

    model_xml_dir = Path(args.model_xml_dir)
    
    save_dir = Path("output") / "demonstration" / model_xml_dir.stem

    generate_data(model_xml_dir,
                  save_dir, 
                  args=args,)
