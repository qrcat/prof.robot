import numpy as np 
from filelock import FileLock
import json
import pickle
import os
import mujoco
import re
import shutil

def get_canonical_pose(joint_limits : np.ndarray, robot_name : str = None) -> np.ndarray:
    """
    Get the canonical pose for the robot.
    Input:
    - joint_limits: np.ndarray, shape [N, 2], where N is the number of joints
    Output:
    - canonical_pose: np.ndarray, shape [N], where N is the number of joints
    """
    
    #[N, 2]
    min_limits = joint_limits[:, 0]
    max_limits = joint_limits[:, 1]

    canonical_pose = np.zeros(joint_limits.shape[0])
    nonzero_joint_indices = np.where((min_limits > 0) | (max_limits < 0))[0]
    
    #make nonzero joints as close to 0 as possible
    for i in nonzero_joint_indices:
        if min_limits[i] > 0:
            print(f"Setting joint {i} to {min_limits[i]}")
            canonical_pose[i] = min_limits[i]
        else:
            print(f"Setting joint {i} to {max_limits[i]}")
            canonical_pose[i] = max_limits[i] 

    if 'franka_emika_panda' in robot_name:
        canonical_pose = np.mean(joint_limits, axis=1)

    return canonical_pose

def set_xml_light_params(model_xml_path, diffuse_light_params, ambient_light_params):
    lock = FileLock(model_xml_path + ".lock")
    with lock:
        with open(model_xml_path, 'r') as file:
            xml_content = file.read()

        # Modify the XML content to set the diffuse and ambient parameters
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_content)
        for visual in root.findall('visual'):
            headlight = visual.find('headlight')
            if headlight is not None:
                headlight.set('diffuse', " ".join([str(x) for x in diffuse_light_params]))  # Set your desired diffuse values
                headlight.set('ambient', " ".join([str(x) for x in ambient_light_params]))
        # Convert the modified XML back to a string
        modified_xml_content = ET.tostring(root, encoding='unicode')

        # Overwrite the original XML file with the modified content using a file lock to ensure process safety
        with open(model_xml_path, 'w') as file:
            file.write(modified_xml_content)

def find_non_collision_pose(joint_limits, 
                            model, 
                            data, 
                            is_canonical, 
                            max_n_collisions=10,
                            robot_name='[Robot]'):
    """
    Robot poses that lead to collisions is suboptimal training data for the 3D Gaussians. 
    Unfortunately, not every sampled pose is collision free. This function tries to find a pose without collision.
    If it can't find no-collision pose fast, it starts looking for poses with 1 collision, then 2 collisions, etc.

    Input:
    - joint_limits: np.ndarray, shape [N, 2], where N is the number of joints
    - model: mujoco.MjModel, the model of the robot
    - data: mujoco.MjData, the data of the robot
    - is_canonical: bool, if the robot is in its canonical pose
    - max_n_collisions: int, the maximum number of collisions to allow
    - robot_name: str, the name of the robot

    Output:
    - joint_position: np.ndarray, shape [N], the joint position of the robot
    """

    find_pose_iters = 0
    n_collisions_allowed = 0 #gets relaxed over time if a good pose cannot be found quickly
    MAX_N_COLLISIONS = 2 if is_canonical else 10
    while True:
        if is_canonical:
            joint_position = get_canonical_pose(joint_limits, robot_name=robot_name)
        else:
            joint_position = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1]) # 随机采样Pose
    
        mujoco.mj_resetData(model, data)
        data.qpos[:] = joint_position
        mujoco.mj_step(model, data)
        mujoco.mj_collision(model, data)

        if (data.ncon <= max_n_collisions):
            break
        break
        find_pose_iters += 1
        if find_pose_iters > 50 * n_collisions_allowed: #more iterations as well
            n_collisions_allowed += 1

        if n_collisions_allowed > MAX_N_COLLISIONS:
            print(f"[{robot_name}] Failed to generate non-collision sample")
            raise Exception(f"[{robot_name}] Failed to generate non-collision sample")
    return joint_position, data.ncon, find_pose_iters

def compute_camera_extrinsic_matrix(cam):
    """Returns the 4x4 extrinsic matrix considering lookat, distance, azimuth, and elevation."""
    
    # Convert azimuth and elevation to radians
    # print("azimuth type: ", type(cam.azimuth))
    # print("azimuth: ", cam.azimuth)
    azimuth_rad = np.deg2rad(round(float(cam.azimuth) + 1e-3))
    elevation_rad = np.deg2rad(round(float(cam.elevation) + 1e-3))
    r = float(cam.distance)

    # Compute the camera position in spherical coordinates (of mujoco)
    x = -r * np.cos(azimuth_rad) * np.cos(elevation_rad)
    y = -r * np.sin(azimuth_rad) * np.cos(elevation_rad)
    z = -r * np.sin(elevation_rad)

    x += cam.lookat[0]
    y += cam.lookat[1]
    z += cam.lookat[2]

    C = np.array([x, y, z, 1])

    # Compute the camera's forward vector
    forward = cam.lookat - C[:3]
    forward = forward / np.linalg.norm(forward)

    # Compute the camera's right vector
    right = np.cross(forward, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right)
    
    # Compute the camera's up vector
    up = np.cross(right, forward)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [-up[0], -up[1], -up[2], 0],
        [forward[0], forward[1], forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Construct the translation matrix
    t = -rotation_matrix @ C
    # Compute the extrinsic matrix

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix[:3, :3]
    extrinsic_matrix[:3, 3] = t[:3]
    
    return extrinsic_matrix

def extract_camera_parameters(extrinsic_matrix, lookat=[0, 0, 0]):
    """
    Extracts camera parameters (azimuth, elevation, distance, lookat) from the extrinsic matrix.
    
    :param extrinsic_matrix: 4x4 camera extrinsic matrix
    :return: Dictionary containing azimuth (degrees), elevation (degrees), distance, and lookat
    """
    # Extract rotation matrix and translation vector
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]
    # Camera position in world coordinates
    C = -R.T @ t
    print("new C: ", C)
    print("new t: ", t)

    # Calculate lookat point (assuming it's the point the camera is facing)
    forward = R[2, :]  # The third row of R represents the forward direction
    print("new forward: ", forward)
    # Calculate distance
    distance = np.linalg.norm(C - lookat)
    print("new distance: ", distance)

    # Calculate azimuth and elevation
    delta = C - lookat
    azimuth = np.arctan2(-delta[1], -delta[0])
    elevation = np.arcsin(-delta[2] / distance)

    # Convert to degrees
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)

    print("new azimuth: ", azimuth_deg)
    print("new elevation: ", elevation_deg)

    return {
        'azimuth': azimuth_deg,
        'elevation': elevation_deg,
        'distance': distance,
    }

def compute_camera_intrinsic_matrix(model, renderer, data):
    """Returns the 3x3 intrinsic matrix."""
    renderer.update_scene(data)
    fov = np.deg2rad(model.vis.global_.fovy)
    # print("fov is ", fov)

    focal_length = 0.5 * renderer.height / np.tan(0.5 * fov)
    aspect_ratio = renderer.width / renderer.height

    intrinsic_matrix = np.array([
        [focal_length, 0, renderer.width / 2],
        [0, focal_length, renderer.height / 2],
        [0, 0, 1]
    ])

    return intrinsic_matrix

def save_robot_metadata(model, model_xml_dir, save_dir):
    """
    Save the robot's metadata to a file e.g. joint limits
    """
    joint_limits = model.jnt_range.copy()
    joint_range_dict = {
        model.jnt(i).name: model.jnt(i).range
        for i in range(model.njnt)
    }
    n_joints = model.njnt
    metadata = {
        'n_joints': n_joints,
        'joint_limits': joint_limits,
        'joint_range_dict': joint_range_dict,
        'model_xml_dir': model_xml_dir
    }
    
    #save as both txt (convert everything to str) and pkl (save as pickle)
    with open(os.path.join(save_dir, "robot_metadata.txt"), "w") as f:
        f.write(str(metadata))
    with open(os.path.join(save_dir, "robot_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    #copy model xml directory into save_dir
    if not os.path.exists(os.path.join(save_dir, 'robot_xml')):
        shutil.copytree(model_xml_dir, os.path.join(save_dir, 'robot_xml'))

def simulate_mujoco_scene(joint_position, 
                          azimuth, 
                          elevation, 
                          distance, 
                          model, 
                          data, 
                          renderer, 
                          lookat=[0, 0, 0], 
                          max_seg_id = 80, 
                          seg=False,
                          unnormalize_joint_angles=False,
                          background_color=[0, 0, 0]):

    mujoco.mj_resetData(model, data)
    joint_limits = model.jnt_range.copy()
    if unnormalize_joint_angles:
        joint_position = (joint_position + 1)/2 * (joint_limits[:, 1] - joint_limits[:, 0]) + joint_limits[:, 0]
    data.qpos[:len(joint_position)] = joint_position
    mujoco.mj_step(model, data)

    # Configure the camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat = np.array(lookat)  # Adjust if necessary

    # Update the scene and render
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()

    # compute extrinsic matrix
    extrinsic_matrix = compute_camera_extrinsic_matrix(cam)

    if seg or background_color:
        # update renderer to render segmentation
        renderer.enable_segmentation_rendering()

        # reset the scene
        renderer.update_scene(data, camera=cam)

        seg = renderer.render()
        seg = (seg[:, :, 0] > 0) * (seg[:, :, 0] < max_seg_id)

        renderer.disable_segmentation_rendering()
    else:
        seg = None 

    if background_color is not None:
        #seg.shape (H, W)
        #pixels.shape (H, W, 3)
        assert len(background_color) == 3 and all(0 <= c <= 1 for c in background_color), f"background color must be a list of 3 numbers between 0 and 1, got {background_color}"
        background_color = np.array(background_color) * 255
        pixels = pixels * seg[:, :, None] + background_color * (1 - seg[:, :, None])
    #check collisions
    mujoco.mj_collision(model, data)

    info = {
        "n_collisions": data.ncon,
        "robot_mask": seg,
    }

    return pixels, extrinsic_matrix, info