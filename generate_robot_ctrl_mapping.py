import numpy as np
import os
import itertools
import ray
import json
import time

from tqdm import tqdm
import glob
import mujoco
import shutil
import pathlib
import matplotlib.pyplot as plt
from random import choice
from PIL import Image
from utils.mujoco_utils import compute_camera_extrinsic_matrix, compute_camera_intrinsic_matrix
from filelock import FileLock
from utils.mujoco_utils import get_canonical_pose, set_xml_light_params, find_non_collision_pose, save_robot_metadata

import copy
import random
import pickle

os.environ['MUJOCO_GL'] = 'egl'

"""
This file creates many Mujoco environments simultaneously and renders images from them. 
It is used to generate the point cloud dataset for the robot dataset.

Instructions for preparing a XML file for processing:
- Remove fixed joints from the model
"""


@ray.remote
class MujocoActor:
    """
    Converts a XML file into an image/depth/point cloud dataset for kinematically-aware gaussian training.
    """
    def __init__(self,
                 actor_id,
                 model_xml_dir, 
                 save_dir,
                 resolution=(256, 256),
                 ):

        self.model_xml_dir = model_xml_dir
        self.model_xml_path = os.path.join(model_xml_dir, "scene.xml")
        self.robot_name = model_xml_dir.split('/')[-1]

        #initiilize the mujoco environment in a fault-tolerant way
        attempt, MAX_ATTEMPTS = 0, 10
        success = False
        while not success and attempt < MAX_ATTEMPTS:
            try:
                self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)
                self.data = mujoco.MjData(self.model)
                self.save_dir = save_dir
                self.renderer = mujoco.Renderer(self.model, resolution[0], resolution[1])
                success = True
            except Exception as e:
                attempt += 1
                sleep_time = 1.5 ** attempt
                print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
        if not success:
            raise Exception("Failed to initialize Mujoco components after multiple attempts.")
        
        save_robot_metadata(self.model, self.model_xml_dir, self.save_dir)


    def sample_ctrl(self, start=None, end=None):
        if start is None and end is None:
            return np.random.uniform(self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]) # 随机采样Pose
        elif start is not None and end is None:
            return np.random.uniform(self.model.actuator_ctrlrange[start:, 0], self.model.actuator_ctrlrange[start:, 1]) # 随机采样Pose
        elif start is None and end is not None:
            return np.random.uniform(self.model.actuator_ctrlrange[:end, 0], self.model.actuator_ctrlrange[:end, 1]) # 随机采样Pose
        elif start is not None and end is not None:
            return np.random.uniform(self.model.actuator_ctrlrange[start:end, 0], self.model.actuator_ctrlrange[start:end, 1]) # 随机采样Pose

    
    def get_uniform_pose(self, num_joints=None) -> np.ndarray:
        if num_joints is None:
            return np.random.uniform(self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]) # 随机采样Pose
        else:
            return np.random.uniform(self.model.jnt_range[:num_joints, 0], self.model.jnt_range[:num_joints, 1]) # 随机采样Pose

    def sample_pose(self, init_pose: np.ndarray, std=10.0, num_joints=None):
        if num_joints is None:
            normal_std = (self.model.jnt_range[:, 1]-self.model.jnt_range[:, 0]) / std
            joint_position: np.ndarray = init_pose + np.random.randn(*init_pose.shape) * normal_std
            joint_position = joint_position.clip(self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]) # clip to limit
        else:
            normal_std = (self.model.jnt_range[:num_joints, 1]-self.model.jnt_range[:num_joints, 0]) / std
            joint_position: np.ndarray = init_pose + np.random.randn(*init_pose.shape) * normal_std
            joint_position = joint_position.clip(self.model.jnt_range[:num_joints, 0], self.model.jnt_range[:num_joints, 1]) # clip to limit
        return joint_position

    def generate_and_save_pc(self, sample_id, args, is_canonical=False, is_test=False, verbose=False, num_joints=6):
        ctrl_list = []
        qpos_list = []

        init_joint = None
        while len(ctrl_list) < 10000:

            ctrl = self.sample_ctrl(start=6)
            self.data.ctrl[6:] = ctrl
            # self.data.qpos[:6] = ctrl[:6]
            mujoco.mj_step(self.model, self.data, 100)

            ctrl_list.append(self.data.ctrl)
            qpos_list.append(self.data.qpos)

        #file saving stuff
        directory_name = f"sample_{sample_id}" 
        unique_dir = os.path.join(self.save_dir, directory_name)
        if not os.path.exists(unique_dir):
            os.makedirs(unique_dir, exist_ok=True)
        # print(directory_name)

        ctrl = np.asarray(ctrl_list)
        qpos = np.asarray(qpos_list)

        with open(os.path.join(unique_dir, 'ctrl.npz'), 'wb') as f:
            np.savez(f, 
                     ctrl=ctrl, 
                     qpos=qpos,
                     )
    
def generate_data(num_actors, num_samples, model_xml_dir, save_dir, args, is_canonical=False, is_test=False, verbose=False): # 产生数据
    actors = [MujocoActor.remote(actor_id, model_xml_dir, save_dir) for actor_id in range(num_actors)]

    tasks = []
    for i in range(num_samples):
        actor_index = i % num_actors
        task = actors[actor_index].generate_and_save_pc.remote(i, args, is_canonical, is_test, verbose)
        tasks.append(task)

    robot_name = os.path.basename(model_xml_dir)
    sample_type = 'canonical' if is_canonical else 'test' if is_test else 'normal'
    pbar = tqdm(total=num_samples, desc=f"Generating {sample_type} data for {robot_name}")
    start_time = time.time()

    while True:
     
        sample_type_prefix = "canonical_" if is_canonical else "test_" if is_test else ""
        num_files = len(glob.glob(os.path.join(save_dir, f"{sample_type_prefix}sample_*")))
        pbar.n = num_files
        pbar.refresh()

        if num_files >= num_samples:
            break

        time.sleep(1)

        elapsed_time = time.time() - start_time
        rate = num_files / elapsed_time
        pbar.set_description(f"Generating {sample_type} data for {robot_name} (Rate: {rate:.2f} files/sec)")

    pbar.close()
    

if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description='Set model XML path and dataset name.')
    parser.add_argument('--model_xml_dir', type=str, default="mujoco_menagerie/universal_robots_ur5e", help='Path to the model XML file.')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset.')
    parser.add_argument('--num_canonical_samples', type=int, default=500, help='Number of canonical samples.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples.')
    parser.add_argument('--num_test', type=int, default=500, help='Number of test samples.')
    parser.add_argument('--num_actors', type=int, default=20, help='Number of actors.')
    parser.add_argument('--camera_distance_factor', type=float, default=1.0, help='Factor to scale the camera distance, change this depending on robot size.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args(
        [
            "--model_xml_dir", "mujoco_demo_control/universal_robots_ur5e_robotiq_empty",
        ]
    )

    model_xml_dir = args.model_xml_dir   
    if not args.dataset_name:
        dataset_name = os.path.basename(model_xml_dir)
    else:
        dataset_name = args.dataset_name

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"./data/{dataset_name}_ctrl"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True) 
    ray.init()

    num_actors = args.num_actors
    num_canonical_samples = args.num_canonical_samples
    num_samples = args.num_samples
    num_test = args.num_test

    if args.debug:
        num_samples = 100
        num_canonical_samples = 12
        num_test = 12
        num_actors = 1

    assert num_samples % num_actors == 0
    assert num_canonical_samples % num_actors == 0
    assert num_test % num_actors == 0
    
    generate_data(num_actors, num_samples, model_xml_dir, save_dir, args=args, is_test=False, is_canonical=False, verbose=args.verbose)
