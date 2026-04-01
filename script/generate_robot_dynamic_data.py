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

import mujoco
import matplotlib.pyplot as plt

from utils.mujoco_utils import save_robot_metadata
from utils.pk_utils import build_chain_from_mjcf_path
os.environ['MUJOCO_GL'] = 'disable'

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
    num_obs_range=(1, 10),
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


@ray.remote(num_cpus=1)
def generate_sample(
    scene_xml,
    sample_id,
    save_dir,
    root_name,
    num_poses=10000,
):
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    used_index = np.arange(model.njnt)
    joint_range = model.jnt_range[used_index]

    def get_uniform_pose():
        return np.random.uniform(
            joint_range[:, 0],
            joint_range[:, 1],
        )

    def resample_pose(init_pose, std=0.1):
        return np.clip(
            init_pose + np.random.normal(0, std, size=init_pose.shape),
            joint_range[:, 0],
            joint_range[:, 1],
        )

    joint_list = []
    colli_list = []

    # --------------------------------
    # Adaptive Sampling Parameters
    # --------------------------------

    target_collision_ratio = 0.5
    max_iterations = num_poses * 5

    p_reset = 0.15
    p_drop_collision = 0.1
    p_drop_no_collision = 0.5

    std = 1.0
    init_joint = None

    iterations = 0
    collision_count = 0
    no_collision_count = 0

    # --------------------------------
    # 自适应采样循环
    # --------------------------------

    while len(joint_list) < num_poses and iterations < max_iterations:

        iterations += 1

        if init_joint is not None and random() < p_reset:
            init_joint = None

        pose = (
            get_uniform_pose()
            if init_joint is None
            else resample_pose(init_joint, std)
        )

        mujoco.mj_resetData(model, data)

        data.qpos[used_index] = pose

        mujoco.mj_step(model, data)
        mujoco.mj_collision(model, data)

        has_collision = data.ncon > 0

        current_ratio = collision_count / max(len(joint_list), 1)

        # --------------------------------
        # 动态采样策略
        # --------------------------------

        if has_collision:

            if current_ratio < target_collision_ratio:
                keep_prob = 1.0 - p_drop_collision
            else:
                keep_prob = p_drop_collision

            init_joint = pose

        else:

            if current_ratio > target_collision_ratio:
                keep_prob = 1.0 - p_drop_no_collision
            else:
                keep_prob = p_drop_no_collision

        if random() > keep_prob:
            continue

        joint_list.append(pose)
        colli_list.append(has_collision)

        if has_collision:
            collision_count += 1
        else:
            no_collision_count += 1

    # --------------------------------
    # 保存数据
    # --------------------------------

    joint = np.asarray(joint_list)
    colls = np.asarray(colli_list)

    final_collision_ratio = collision_count / len(joint_list) if len(joint_list) > 0 else 0
    print(f"Sample {sample_id}: Collision ratio = {final_collision_ratio:.3f} " f"(Collision: {collision_count}, No Collision: {no_collision_count})")

    out_dir = os.path.join(
        save_dir,
        f"sample_{sample_id:05d}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # statistics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pie( [collision_count, no_collision_count], labels=["Collision", "No Collision"], autopct='%1.1f%%' )
    plt.title(f"Collision Distribution\nRatio: {final_collision_ratio:.3f}")
    plt.subplot(1, 2, 2)
    plt.bar(["Collision", "No Collision"], [collision_count, no_collision_count])
    plt.title("Sample Counts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "balance_analysis.jpg")) 
    plt.close()

    np.savez(
        os.path.join(out_dir, "data.npz"),
        joint=joint,
        collision=colls,
    )

    mujoco.mj_forward(model, data)

    pc = generate_obstacle_pointcloud(
        model,
        data,
        1000
    )

    if pc is not None:
        save_ply_xyz(
            os.path.join(out_dir, "obstacles.ply"),
            pc
        )

    return True

def generate_data(
    num_workers,
    num_samples,
    scene_dir,
    save_dir,
    root_name,
):
    """Generate collision data using Ray actors."""
    scene_xmls = sorted(
        Path(scene_dir).glob("scene_*.xml")
    )

    running = []

    pbar = tqdm(total=num_samples)

    for i in range(num_samples):

        scene = str(
            scene_xmls[i % len(scene_xmls)]
        )

        task = generate_sample.remote(
            scene,
            i,
            save_dir,
            root_name,
        )

        running.append(task)

        if len(running) >= num_workers * 2:

            done, running = ray.wait(
                running,
                num_returns=1
            )

            pbar.update(1)

    while running:

        done, running = ray.wait(
            running,
            num_returns=1
        )

        pbar.update(1)

    pbar.close()

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
    parser.add_argument("--num_workers", type=int, default=8)
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
    ray.init()

    generate_data(
        args.num_workers,
        args.num_samples,
        args.model_xml_dir,
        save_dir,
        args.root_name,
    )
