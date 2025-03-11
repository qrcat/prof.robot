import numpy as np
import os
import ray
import time

from tqdm import tqdm

import mujoco
import matplotlib.pyplot as plt
from utils.mujoco_utils import save_robot_metadata
from utils.pk_utils import build_chain_from_mjcf_path

from multiprocessing.pool import ThreadPool

import glob
import time
import shutil
import pathlib
from random import random

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
                 args,
                 ):

        self.model_xml_dir = model_xml_dir
        self.model_xml_path = os.path.join(model_xml_dir, "scene.xml")
        self.robot_name = model_xml_dir.split('/')[-1]
        
        #initiilize the mujoco environment in a fault-tolerant way
        attempt, MAX_ATTEMPTS = 0, 10
        success = False
        while not success and attempt < MAX_ATTEMPTS:
            try:
                self.chain = build_chain_from_mjcf_path(self.model_xml_path, args.root_name)
                self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)
                self.data = mujoco.MjData(self.model)

                self.save_dir = save_dir
                success = True
            except Exception as e:
                print(e)
                attempt += 1
                sleep_time = 1.5 ** attempt
                # print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
        if not success:
            raise Exception("Failed to initialize Mujoco components after multiple attempts.")

        num_joints = self.model.njnt
        
        model_joints = {}

        for i in range(num_joints):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            
            qpos_index = self.model.jnt_qposadr[i]

            model_joints[joint_name] = qpos_index

        if False: # scene info
            self.used_index = np.asarray([model_joints[joint] for joint in self.chain.get_joint_parameter_names()])
        else:
            self.used_index = np.arange(num_joints)

        self.joint_range = self.model.jnt_range[self.used_index]

        save_robot_metadata(self.model, self.model_xml_dir, self.save_dir)

    def get_uniform_pose(self) -> np.ndarray:
        return np.random.uniform(self.joint_range[:, 0], self.joint_range[:, 1])
    
    def resample_pose(self, init_pose: np.ndarray, std=0.1,) -> np.ndarray:
        return np.clip(init_pose + np.random.normal(0, std, size=init_pose.shape), self.joint_range[:, 0], self.joint_range[:, 1])

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
        joint_list = []
        colli_list = []

        p_reset = 1.0
        p_drop = 0.0
        std = 1.0

        init_joint = None
        while len(joint_list) < 100000:
            if init_joint is not None and random() < p_reset: init_joint = None

            if init_joint is None:
                pose = self.get_uniform_pose()
            else:
                pose = self.resample_pose(init_joint, std)

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

        self.used_index

        #file saving stuff
        directory_name = f"sample_{sample_id}" 
        if is_canonical:
            directory_name = "canonical_" + directory_name
        if is_test:
            directory_name = "test_" + directory_name
        unique_dir = os.path.join(self.save_dir, directory_name)

        # Use FileLock to ensure only one actor creates the directory

        if not os.path.exists(unique_dir):
            os.makedirs(unique_dir, exist_ok=True)

        joint = np.asarray(joint_list)
        colls = np.asarray(colli_list)

        non_coll_num = (colls==0).sum()
        coll_num = len(colls) - non_coll_num
        plt.figure()
        plt.pie([coll_num, non_coll_num], labels=['Collision', 'No Collision'])
        plt.legend()
        plt.savefig(os.path.join(unique_dir, 'pie.jpg'))
        plt.close()

        with open(os.path.join(unique_dir, 'data.npz'), 'wb') as f:
            np.savez(
                f, 
                joint=joint, 
                collision=colls,
            )
    
def generate_data(num_actors, num_samples, model_xml_dir, save_dir, args, is_canonical=False, is_test=False, verbose=False): # 产生数据
    actors = [MujocoActor.remote(actor_id, model_xml_dir, save_dir, args) for actor_id in range(num_actors)]

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

    data_path = pathlib.Path(save_dir)
    npz_files = list(data_path.glob("sample_*/data.npz"))
    tbar = tqdm(npz_files, "Read NPZ")

    def get_data(path: pathlib.Path, tbar):
        collision_l = []
        collision_n = []

        data = np.load(path)

        collision_l = data["joint"]
        collision_n = data["collision"]

        tbar.update(1)

        return collision_l, collision_n

    with ThreadPool() as pool:
        OUT = pool.map(lambda x: (get_data(x, tbar)), npz_files)
        pool.close()
        pool.join()

    collision_j = np.concatenate([x[0] for x in OUT], 0)
    collision_n = np.concatenate([x[1] for x in OUT], 0)

    joint_data_path = data_path / "joint_data.npz"
    print("save joint data", joint_data_path)
    np.savez(joint_data_path, joint=collision_j, nums=collision_n)



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
    parser.add_argument('--root_name', type=str, default='base', help='Root name of the robot.')
    args = parser.parse_args(
        [
            # "--model_xml_dir", "collision_scene/xMate_SR3",
            # "--root_name", "xMateSR3_base"

            '--model_xml_dir', 'collision_scene/universal_robots_ur5e_scene3',
            '--root_name', 'base',
        ]
    )

    model_xml_dir = args.model_xml_dir   
    if not args.dataset_name:
        dataset_name = os.path.basename(model_xml_dir)
    else:
        dataset_name = args.dataset_name

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"./data/{dataset_name}_collision"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving data to {save_dir}")


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

