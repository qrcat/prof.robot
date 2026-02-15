import numpy as np
import os
import ray
import time
import glob
import shutil
import pathlib

from tqdm import tqdm
from random import random
from pathlib import Path
from multiprocessing.pool import ThreadPool

import mujoco
import matplotlib.pyplot as plt

from utils.mujoco_utils import save_robot_metadata
from utils.pk_utils import build_chain_from_mjcf_path


def save_ply_xyz(path, points):
    """
    Save point cloud to ASCII PLY.
    points: (N, 3)
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def sample_box(size, n=500):
    pts = np.random.uniform(-size, size, size=(n * 3, 3))
    axis = np.random.randint(0, 3, size=n * 3)
    sign = np.random.choice([-1, 1], size=n * 3)
    for i in range(n * 3):
        pts[i, axis[i]] = sign[i] * size[axis[i]]
    return pts[:n]


def sample_sphere(radius, n=500):
    v = np.random.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def sample_capsule(radius, half_height, n=500):
    pts = []
    for _ in range(n):
        if np.random.random() < 0.5:
            theta = np.random.random() * 2 * np.pi
            z = np.random.uniform(-half_height, half_height)
            pts.append([radius * np.cos(theta), radius * np.sin(theta), z])
        else:
            v = np.random.normal(size=3)
            v /= np.linalg.norm(v)
            v[2] += half_height if v[2] > 0 else -half_height
            pts.append(v * radius)
    return np.asarray(pts)


def geom_to_points(model, data, geom_id, n_points=800):
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]

    # ✅ world pose (ONLY in data)
    pos = data.geom_xpos[geom_id]  # (3,)
    mat = data.geom_xmat[geom_id].reshape(3, 3)  # (3,3)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        pts = sample_box(size, n_points)
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts = sample_sphere(size[0], n_points)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        pts = sample_capsule(size[0], size[1], n_points)
    else:
        return None

    # local → world
    pts = pts @ mat.T + pos
    return pts


def generate_obstacle_pointcloud(model, data, total_points=-1):
    """Generate obstacle point cloud with specified total number of points."""
    obstacle_points = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if body_name and "obstacle_" in body_name:
            pts = geom_to_points(model, data, geom_id, n_points=800)
            if pts is not None:
                obstacle_points.append(pts)

    if obstacle_points:
        all_obstacle_points = np.vstack(obstacle_points)

        # Downsample to exactly total_points if needed
        if len(all_obstacle_points) > total_points > 0:
            indices = np.random.choice(
                len(all_obstacle_points), total_points, replace=False
            )
            all_obstacle_points = all_obstacle_points[indices]
        return all_obstacle_points
    return None


def make_obstacle_xml(
    idx,
    geom_type="box",
    size=(0.05, 0.05, 0.05),
    pos=(0.5, 0.0, 0.3),
    rgba=(0.7, 0.2, 0.2, 1.0),
    dynamic=True,
):
    joint = '<joint type="free"/>' if dynamic else ""
    size_str = " ".join(map(str, size))
    pos_str = " ".join(map(str, pos))
    rgba_str = " ".join(map(str, rgba))

    return f"""
    <body name="obstacle_{idx}" pos="{pos_str}">
      {joint}
      <geom
        type="{geom_type}"
        size="{size_str}"
        rgba="{rgba_str}"
        contype="1"
        conaffinity="1"
      />
    </body>
    """


def generate_obstacles_xml(num_obs, workspace=((0.3, -0.4, 0.1), (0.8, 0.4, 0.6))):
    obs_xml = []
    for i in range(num_obs):
        geom_type = np.random.choice(["box", "sphere", "capsule"])
        size = np.random.uniform(0.03, 0.08, size=3)
        pos = np.random.uniform(workspace[0], workspace[1])
        obs_xml.append(make_obstacle_xml(i, geom_type, size, pos, dynamic=False))
    return "\n".join(obs_xml)


def copy_meta(
    base_scene_template_path,
    output_dir,
    save_dir,
):
    # Get the template directory from the scene.xml path
    template_dir = os.path.dirname(base_scene_template_path)

    # Copy template directory contents to output directory
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(template_dir):
        src = os.path.join(template_dir, item)
        dst = os.path.join(output_dir, item)
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    with open(base_scene_template_path, "r") as f:
        template_content = f.read()

    # Create scene_no_obstacles.xml (scene without obstacles)
    scene_no_obstacles_path = os.path.join(output_dir, "scene.xml")
    scene_no_obstacles_xml = template_content.replace(
        "{{OBSTACLES}}", "<!-- No obstacles -->"
    )
    with open(scene_no_obstacles_path, "w") as f:
        f.write(scene_no_obstacles_xml)
    print(f"Created {scene_no_obstacles_path}")

    base_xml = Path(output_dir) / "scene.xml"
    model = mujoco.MjModel.from_xml_path(base_xml.as_posix())
    save_robot_metadata(model, output_dir, save_dir)


def generate_scene_xmls(
    base_scene_template_path,
    output_dir,
    num_scenes,
    num_obs_range=(1, 20),
):
    """Generate scene XML files with obstacles."""
    with open(base_scene_template_path, "r") as f:
        template_content = f.read()

    for i in range(num_scenes):
        n_obs = np.random.randint(*num_obs_range)
        obs_xml = generate_obstacles_xml(n_obs)

        scene_xml = template_content.replace("{{OBSTACLES}}", obs_xml)

        scene_path = os.path.join(output_dir, f"scene_{i:04d}.xml")
        with open(scene_path, "w") as f:
            f.write(scene_xml)

    print(f"Created {num_scenes} scene XML files")


@ray.remote
class MujocoActor:
    """
    One actor can process multiple scene XMLs sequentially.
    """

    def __init__(self, actor_id, save_dir, args):
        self.actor_id = actor_id
        self.save_dir = save_dir
        self.args = args
        self.model = None
        self.data = None
        self.chain = None

    def load_scene(self, scene_xml_path):
        """Load a new Mujoco scene"""
        attempt, MAX_ATTEMPTS = 0, 5
        success = False

        while not success and attempt < MAX_ATTEMPTS:
            try:
                self.chain = build_chain_from_mjcf_path(
                    scene_xml_path, self.args.root_name
                )
                self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
                self.data = mujoco.MjData(self.model)

                self.used_index = np.arange(self.model.njnt)
                self.joint_range = self.model.jnt_range[self.used_index]

                success = True
            except Exception as e:
                attempt += 1
                time.sleep(1.5**attempt)

        if not success:
            raise RuntimeError(f"Failed to load scene {scene_xml_path}")

    def get_uniform_pose(self):
        return np.random.uniform(self.joint_range[:, 0], self.joint_range[:, 1])

    def resample_pose(self, init_pose, std=0.1):
        return np.clip(
            init_pose + np.random.normal(0, std, size=init_pose.shape),
            self.joint_range[:, 0],
            self.joint_range[:, 1],
        )

    def generate_and_save_pc(
        self,
        scene_xml_path,
        sample_id,
        num_poses=10000,
        is_canonical=False,
        is_test=False,
    ):
        self.load_scene(scene_xml_path)

        joint_list = []
        colli_list = []

        p_reset = 1.0
        p_drop = 0.5
        std = 1.0
        init_joint = None

        while len(joint_list) < num_poses:
            if init_joint is not None and random() < p_reset:
                init_joint = None

            pose = (
                self.get_uniform_pose()
                if init_joint is None
                else self.resample_pose(init_joint, std)
            )

            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.used_index] = pose
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_collision(self.model, self.data)

            if self.data.ncon > 0:
                init_joint = pose
            elif random() < p_drop:
                continue

            joint_list.append(pose)
            colli_list.append(self.data.ncon)

        prefix = "canonical_" if is_canonical else "test_" if is_test else ""
        out_dir = os.path.join(self.save_dir, f"{prefix}sample_{sample_id:05d}")
        os.makedirs(out_dir, exist_ok=True)

        joint = np.asarray(joint_list)
        colls = np.asarray(colli_list)

        # statistics
        plt.figure()
        plt.pie(
            [(colls > 0).sum(), (colls == 0).sum()],
            labels=["Collision", "No Collision"],
        )
        plt.savefig(os.path.join(out_dir, "pie.jpg"))
        plt.close()

        np.savez(
            os.path.join(out_dir, "data.npz"),
            joint=joint,
            collision=colls,
            scene_xml=os.path.basename(scene_xml_path),
        )

        # Extract and save obstacle point cloud
        mujoco.mj_forward(self.model, self.data)
        obstacle_pc = generate_obstacle_pointcloud(
            self.model,
            self.data,
        )
        if obstacle_pc is not None:
            save_ply_xyz(os.path.join(out_dir, "obstacles.ply"), obstacle_pc)
            print(f"Saved obstacle point cloud with {len(obstacle_pc)} points")


def generate_data(
    num_actors,
    num_samples,
    scene_xml_dir,
    save_dir,
    args,
    is_canonical=False,
    is_test=False,
):
    """Generate collision data using Ray actors."""
    scene_xmls = sorted(Path(scene_xml_dir).glob("scene_*.xml"))
    assert len(scene_xmls) > 0, "No scene XML files found"

    actors = [MujocoActor.remote(i, save_dir, args) for i in range(num_actors)]

    tasks = []
    for i in range(num_samples):
        actor = actors[i % num_actors]
        scene_xml = scene_xmls[i % len(scene_xmls)]
        task = actor.generate_and_save_pc.remote(
            str(scene_xml),
            sample_id=i,
            is_canonical=is_canonical,
            is_test=is_test,
        )
        tasks.append(task)

    pbar = tqdm(total=num_samples, desc="Generating collision data")
    start = time.time()

    while True:
        prefix = "canonical_" if is_canonical else "test_" if is_test else ""
        done = len(glob.glob(os.path.join(save_dir, f"{prefix}sample_*")))
        pbar.n = done
        pbar.refresh()
        if done >= num_samples:
            break
        time.sleep(1)

    pbar.close()
    ray.get(tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_xml",
        type=str,
        default="mujoco_menagerie/universal_robots_ur5e_dynamic/scene.xml",
    )
    parser.add_argument(
        "--model_xml_dir", type=str, default="data/universal_robots_ur5e_dynamic"
    )
    parser.add_argument("--num_scenes", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_actors", type=int, default=8)
    parser.add_argument("--root_name", type=str, default="base")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Step 0: Copy Template Datasets
    save_dir = f"./data/{Path(args.model_xml_dir).name}_collision"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    copy_meta(
        args.template_xml,
        args.model_xml_dir,
        save_dir,
    )

    # Step 1: Generate XML files
    print("Step 1: Generating scene XML files...")
    generate_scene_xmls(
        args.template_xml,
        args.model_xml_dir,
        args.num_scenes,
    )

    # Step 2: Generate data using Ray
    print("\nStep 2: Generating collision data using Ray...")
    ray.init(ignore_reinit_error=True)

    generate_data(
        num_actors=args.num_actors,
        num_samples=args.num_samples,
        scene_xml_dir=args.model_xml_dir,
        save_dir=save_dir,
        args=args,
    )
