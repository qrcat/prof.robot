import sys
import tempfile

"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_reconstruct.py --model_path output/[path_to_your_model_directory]
"""


INSTRUCTIONS = """
# Optimization Workflow

## 1. Simple Joint Optimization 
To test a simple joint optimization:
1. Click "Random Initialize Joints" for MuJoCo to set random joint angles.
2. Click "Copy MuJoCo Joints" to transfer these parameters to the Gaussian renderer.
3. Slightly adjust one of the Gaussian joint angles (e.g., change θ1 by 0.1).
4. Check "Optimize Joints" and uncheck "Optimize Camera Parameters".
5. Click "Optimize" to start the optimization process.

## 2. Camera Optimization
To test camera parameter optimization:
1. Click "Random Initialize Camera" for MuJoCo to set random camera parameters.
2. Ensure the joint angles are the same for both MuJoCo and Gaussian renderer.
3. Check "Optimize Camera Parameters" and uncheck "Optimize Joints".
4. Click "Optimize" to start the optimization process.

## 3. Joint and Camera Optimization
To test both joint and camera optimization:
1. Use "Random Initialize Joints" and "Random Initialize Camera" for MuJoCo.
2. Check both "Optimize Joints" and "Optimize Camera Parameters".
3. Set appropriate learning rates and optimization steps.
4. Click "Optimize" to start the combined optimization process.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir

import gradio as gr

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

import numpy as np
import mujoco
from utils.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from tqdm import tqdm

from video_api import initialize_gaussians
from gaussian_renderer import render
from scene.cameras import Camera_Pose, Camera
import torch
import torch.nn.functional as F

gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
background_color = torch.zeros((3,)).cuda()
example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'scene.xml'))
data = mujoco.MjData(model)
n_joints = model.njnt

def render_scene(*args):
    os.environ['MUJOCO_GL'] = 'egl'
    
    n_params = len(args)
    n_joints = n_params - 3
    
    joint_angles = np.array(args[:n_joints])
    azimuth, elevation, distance = args[n_joints:]
    lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]
    
    renderer = mujoco.Renderer(model, 480, 480)
    
    pixels, _, _ = simulate_mujoco_scene(joint_angles, 
                                         azimuth, 
                                         elevation, 
                                         distance, 
                                         model, 
                                         data, 
                                         renderer,
                                         lookat,
                                         unnormalize_joint_angles=True)
    
    del renderer
    
    return pixels

def reset_params():
    return [0] * n_joints + [0, -45, 2]

class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*args):
    n_params = len(args)
    n_joints = n_params - 3
    
    joint_angles = torch.tensor(args[:n_joints])
    azimuth, elevation, distance = args[n_joints:]

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), example_camera.FoVx, example_camera.FoVy,\
                            480, 480, joint_pose=joint_angles, zero_init=True).cuda()
    frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def gaussian_reset_params():
    return [0] * n_joints + [0, -45, 2]

def random_initialize_joints():
    return [round(np.random.uniform(-1, 1), 2) for _ in range(n_joints)]

def random_initialize_camera():
    random_azimuth = round(np.random.uniform(0, 360), 2)
    random_elevation = round(np.random.uniform(-90, 90), 2)
    random_distance = round(np.random.uniform(1, 3), 2)
    return [random_azimuth, random_elevation, random_distance]

def random_initialize():
    return random_initialize_joints() + random_initialize_camera()

def gaussian_random_initialize_joints():
    return random_initialize_joints()

def gaussian_random_initialize_camera():
    return random_initialize_camera()

def copy_mujoco_camera(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[n_joints:]

def copy_mujoco_joints(*mujoco_params):
    n_joints = len(mujoco_joint_inputs)
    return mujoco_params[:n_joints]

def initial_render():
    initial_params = reset_params()
    mujoco_image = render_scene(*initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    return mujoco_image, gaussian_image

def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, powerful_optimize_dropdown, noise_input, num_inits_input, *all_params):
    n_params = len(all_params) // 2
    mujoco_params = all_params[:n_params]
    gaussian_params = list(all_params[n_params:])

    mujoco_image = render_scene(*mujoco_params)
    mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1)

    n_params = len(gaussian_params)
    n_joints = n_params - 3

    if powerful_optimize_dropdown == "Enabled":
        num_inits = int(num_inits_input)
        grid_size = int(np.sqrt(num_inits))
    else:
        num_inits = 1
        grid_size = 1
        noise_input = 0

    all_cameras = []
    all_joint_poses = []

    for init in range(num_inits):
        # Only perturb parameters that are being optimized
        print("gaussian_params", gaussian_params)
        perturbed_params = gaussian_params.copy()
        
        if optimize_joints:
            perturbed_params[:n_joints] = [float(p) + np.random.normal(0, noise_input) for p in perturbed_params[:n_joints]]
        
        joint_angles = torch.tensor(perturbed_params[:n_joints], dtype=torch.float32, requires_grad=optimize_joints)
        
        if optimize_camera:
            perturbed_params[n_joints:] = [float(p) + np.random.normal(0, noise_input) for p in perturbed_params[n_joints:]]
        
        azimuth, elevation, distance = perturbed_params[n_joints:]

        dummy_cam = DummyCam(azimuth, elevation, distance)
        camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), example_camera.FoVx, example_camera.FoVy,\
                                480, 480, joint_pose=joint_angles, zero_init=True).cuda()

        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)

    optimizers = []
    if optimize_camera:
        optimizers.append(torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr))
    if optimize_joints:
        optimizers.append(torch.optim.Adam(all_joint_poses, lr=joints_lr))

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        for optimizer in optimizers:
            optimizer.zero_grad()

        gaussian_tensors = []
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            camera.joint_pose = joint_pose
            gaussian_tensors.append(render(camera, gaussians, background_color)['render'])
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        
        mse_loss = F.mse_loss(mujoco_tensor.unsqueeze(0).expand_as(gaussian_tensors), gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3))
        
        total_loss = l2_diff.sum()
        total_loss.backward()

        all_losses = l2_diff.detach().cpu().numpy().tolist()
        all_images = [torch.clamp(tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() for tensor in gaussian_tensors]

        for optimizer in optimizers:
            optimizer.step()

        if step % 10 == 0 or step == optimization_steps - 1:
            grid_images = np.zeros((480 * grid_size, 480 * grid_size, 3))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < len(all_images):
                        grid_images[i*480:(i+1)*480, j*480:(j+1)*480] = all_images[idx]

            best_idx = np.argmin(all_losses)
            best_camera = all_cameras[best_idx]
            best_joint_pose = all_joint_poses[best_idx]
            
            # Extract updated parameters for the best camera
            updated_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
            updated_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
            updated_camera_params = extract_camera_parameters(updated_extrinsic.T)
            
            updated_params = updated_joint_angles + [
                updated_camera_params['azimuth'],
                updated_camera_params['elevation'],
                updated_camera_params['distance']
            ]

            rounded_params = [round(param, 2) for param in updated_params]

            yield (grid_images, *rounded_params)

    # Final yield with only the best result
    best_idx = np.argmin(all_losses)
    best_image = all_images[best_idx]
    best_camera = all_cameras[best_idx]
    best_joint_pose = all_joint_poses[best_idx]

    # Extract final parameters for the best camera
    final_joint_angles = best_joint_pose.detach().cpu().numpy().tolist()
    final_extrinsic = best_camera.world_view_transform.detach().cpu().numpy()
    final_camera_params = extract_camera_parameters(final_extrinsic.T)
    
    final_params = final_joint_angles + [
        final_camera_params['azimuth'],
        final_camera_params['elevation'],
        final_camera_params['distance']
    ]

    rounded_final_params = [round(param, 2) for param in final_params]

    yield (best_image, *rounded_final_params)

with gr.Blocks() as demo:
    
    with gr.Row():
        mujoco_output_image = gr.Image(type="numpy", label="MuJoCo Rendered Scene")
        gaussian_output_image = gr.Image(type="numpy", label="Gaussian Rendered Scene")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## MuJoCo Renderer")
            with gr.Row():
                mujoco_joint_inputs = [gr.Number(label=f"θ{i+1}", value=0) for i in range(n_joints)]
            
            with gr.Row():
                mujoco_camera_inputs = [
                    gr.Number(label="Azimuth (deg)", value=0),
                    gr.Number(label="Elevation (deg)", value=-45),
                    gr.Number(label="Distance (m)", value=2),
                ]
            
            with gr.Row():
                mujoco_render_button = gr.Button("Render MuJoCo", scale=1)
                mujoco_reset_button = gr.Button("Reset MuJoCo", scale=1)
                mujoco_random_joints_button = gr.Button("Random Initialize Joints", scale=1)
                mujoco_random_camera_button = gr.Button("Random Initialize Camera", scale=1)
            
            mujoco_all_inputs = mujoco_joint_inputs + mujoco_camera_inputs
            mujoco_render_button.click(fn=render_scene, inputs=mujoco_all_inputs, outputs=mujoco_output_image)
            mujoco_reset_button.click(fn=reset_params, outputs=mujoco_all_inputs)
            mujoco_random_joints_button.click(fn=random_initialize_joints, outputs=mujoco_joint_inputs)
            mujoco_random_camera_button.click(fn=random_initialize_camera, outputs=mujoco_camera_inputs)
        
        with gr.Column():
            gr.Markdown("## Gaussian Renderer")
            with gr.Row():
                gaussian_joint_inputs = [gr.Number(label=f"θ{i+1}", value=0) for i in range(n_joints)]
            
            with gr.Row():
                gaussian_camera_inputs = [
                    gr.Number(label="Azimuth (deg)", value=0),
                    gr.Number(label="Elevation (deg)", value=-45),
                    gr.Number(label="Distance (m)", value=2),
                ]
            
            with gr.Row():
                gaussian_render_button = gr.Button("Render Gaussian", scale=1)
                gaussian_reset_button = gr.Button("Reset Gaussian", scale=1)
                gaussian_random_joints_button = gr.Button("Random Initialize Joints", scale=1)
                gaussian_random_camera_button = gr.Button("Random Initialize Camera", scale=1)
            
            with gr.Row():
                gaussian_copy_camera_button = gr.Button("Copy MuJoCo Camera", scale=1)
                gaussian_copy_joints_button = gr.Button("Copy MuJoCo Joints", scale=1)
            
            with gr.Row():
                with gr.Column(scale=1):
                    optimize_camera = gr.Checkbox(label="Optimize Camera Parameters", value=False)
                    optimize_joints = gr.Checkbox(label="Optimize Joints", value=False)
                with gr.Column(scale=1):
                    camera_lr = gr.Number(label="Camera Learning Rate", value=0.02)
                    joints_lr = gr.Number(label="Joints Learning Rate", value=0.02)
                with gr.Column(scale=1):
                    optimization_steps = gr.Number(label="Optimization Steps", value=50, step=1)
                    optimize_button = gr.Button("Optimize", scale=1)
            
            with gr.Row():
                powerful_optimize_dropdown = gr.Dropdown(
                    label="Powerful Optimize",
                    choices=["Disabled", "Enabled"],
                    value="Disabled"
                )
            
            with gr.Row():
                noise_input = gr.Number(
                    label="Insert Noise Amount",
                    value=0.01,
                    step=0.001,
                    precision=3,
                    visible=False
                )
                num_inits_input = gr.Dropdown(
                    label="Number of Initializations",
                    choices=["1", "4", "9", "16"],
                    value="1",
                    visible=False
                )
            
            gaussian_all_inputs = gaussian_joint_inputs + gaussian_camera_inputs
            gaussian_render_button.click(fn=gaussian_render_scene, inputs=gaussian_all_inputs, outputs=gaussian_output_image)
            gaussian_reset_button.click(fn=gaussian_reset_params, outputs=gaussian_all_inputs)
            gaussian_random_joints_button.click(fn=gaussian_random_initialize_joints, outputs=gaussian_joint_inputs)
            gaussian_random_camera_button.click(fn=gaussian_random_initialize_camera, outputs=gaussian_camera_inputs)
            
            gaussian_copy_camera_button.click(
                fn=copy_mujoco_camera,
                inputs=mujoco_all_inputs,
                outputs=gaussian_camera_inputs
            )
            
            gaussian_copy_joints_button.click(
                fn=copy_mujoco_joints,
                inputs=mujoco_all_inputs,
                outputs=gaussian_joint_inputs
            )
            
            optimize_button.click(
                fn=optimize,
                inputs=[optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        powerful_optimize_dropdown, noise_input, num_inits_input] + 
                    mujoco_all_inputs + gaussian_all_inputs,
                outputs=[gaussian_output_image] + gaussian_all_inputs,
                show_progress=True
            )
        
    gr.Markdown("# Robot Scene Renderer Comparison")
    gr.Markdown(INSTRUCTIONS)   
    
    demo.load(fn=initial_render, outputs=[mujoco_output_image, gaussian_output_image])

    # Move this inside the gr.Blocks() context
    powerful_optimize_dropdown.change(
        fn=lambda x: [gr.update(visible=(x == "Enabled"))] * 3,
        inputs=[powerful_optimize_dropdown],
        outputs=[noise_input, num_inits_input]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=8900)