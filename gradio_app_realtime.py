"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_realtime.py --model_path output/[path_to_your_model_directory]
"""


import sys
import tempfile

import os
os.environ['MUJOCO_GL'] = 'egl'

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir
import queue
import threading

import gradio as gr
from utils.mujoco_utils import compute_camera_extrinsic_matrix, extract_camera_parameters

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
import time

gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
background_color_list = [0, 0, 0]
background_color = torch.tensor(background_color_list).float().cuda()
example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'scene.xml'))
data = mujoco.MjData(model)
n_joints = model.njnt
import os
import tempfile
import gradio as gr
import numpy as np
import mujoco
from utils.mujoco_utils import simulate_mujoco_scene

class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*joint_angles):
    azimuth, elevation, distance = 0, -45, 3  # Fixed camera parameters
    
    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    joint_angles = torch.tensor(joint_angles)
    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), example_camera.FoVx, example_camera.FoVy,\
                            480, 480, joint_pose=joint_angles, zero_init=True).cuda()
    frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def reset_params():
    new_input_received.set()
    return [0.0] * n_joints

def initial_render():
    initial_params = reset_params()
    print("initial_params: ", initial_params, "rendering scene...")
    mujoco_image = render_scene(*initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    print("done rendering scene")

    optimization_queue.put((initial_params, initial_params))

    return mujoco_image, gaussian_image

def render_scene(*args):
    os.environ['MUJOCO_GL'] = 'egl'
    
    n_params = len(args)
    n_joints = n_params
    
    joint_angles = np.array([float(t) for t in args[:n_joints]])
    azimuth, elevation, distance = [0, -45, 3] #hardcoded
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
                                         unnormalize_joint_angles=True,
                                         background_color=background_color_list)
    
    del renderer
    
    return pixels

# Global variables for optimization
optimization_queue = queue.Queue()
stop_optimization = threading.Event()
new_input_received = threading.Event()

# Add a new global variable
reset_to_mujoco = threading.Event()

def continuous_optimization():
    print("Continuous optimization thread started")
    last_optimized_params = None
    last_update_time = 0
    update_interval = 0.2  # Update every 100ms
    
    while not stop_optimization.is_set():
        try:
            params = optimization_queue.get(timeout=1)
            if params is not None:
                mujoco_params, gaussian_params = params
                print("Starting new optimization with params:", mujoco_params)
                new_input_received.clear()  # Clear the flag before starting optimization
                
                # Use mujoco_params if reset_to_mujoco is set, otherwise use last_optimized_params or gaussian_params
                if reset_to_mujoco.is_set():
                    start_params = mujoco_params
                    reset_to_mujoco.clear()  # Clear the flag after using it
                else:
                    start_params = last_optimized_params if last_optimized_params is not None else gaussian_params
                
                for result, optimized_params in optimize(mujoco_params, start_params, initial_lr=0.02):
                    if new_input_received.is_set():
                        print("New input received, restarting optimization")
                        break
                    last_optimized_params = optimized_params  # Update last_optimized_params
                    
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        print("yielding!")
                        yield result, optimized_params
                        last_update_time = current_time
        except queue.Empty:
            print("Optimization queue is empty")
    print("Continuous optimization thread stopped")

def optimize(mujoco_params, gaussian_params, initial_lr=0.02, decay_factor=0.95, decay_steps=50):
    print("Starting optimization with params:", mujoco_params)
    
    mujoco_image = render_scene(*mujoco_params)
    mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1)

    joint_angles = torch.tensor(gaussian_params, dtype=torch.float32, requires_grad=True)
    azimuth, elevation, distance = 0, -45, 3  # Fixed camera parameters

    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), example_camera.FoVx, example_camera.FoVy,
                         480, 480, joint_pose=joint_angles, zero_init=True).cuda()

    optimizer = torch.optim.Adam([joint_angles], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    # New variables for tracking optimization progress
    previous_loss = float('inf')
    stagnant_iterations = 0
    max_stagnant_iterations = 10
    loss_threshold = 1e-3
    iteration = 0
    N = 2  # Yield every N iterations

    while True:
        optimizer.zero_grad()

        camera.joint_pose = joint_angles
        gaussian_tensor = render(camera, gaussians, background_color)['render']

        loss = F.mse_loss(mujoco_tensor, gaussian_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        print(f"Iteration: {iteration}, Loss: {current_loss}, LR: {scheduler.get_last_lr()[0]:.5f}")

        # Check if the loss has stagnated
        if abs(current_loss - previous_loss) < loss_threshold:
            stagnant_iterations += 1
        else:
            stagnant_iterations = 0

        previous_loss = current_loss
        iteration += 1

        # Yield every N iterations or if it's the last iteration
        if iteration % N == 0 or stagnant_iterations >= max_stagnant_iterations:
            yield torch.clamp(gaussian_tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy(), joint_angles.detach().cpu().numpy()

        # Stop optimization if loss has stagnated for too long
        if stagnant_iterations >= max_stagnant_iterations:
            # print(f"Optimization stopped: Loss stagnated for {max_stagnant_iterations} iterations")
            break

        if stop_optimization.is_set() or new_input_received.is_set():
            # Yield the last image before breaking
            yield torch.clamp(gaussian_tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy(), joint_angles.detach().cpu().numpy()
            break

    print("Optimization finished")

def start_optimization_thread():
    optimization_thread = threading.Thread(target=continuous_optimization)
    optimization_thread.start()
    return optimization_thread

with gr.Blocks() as demo:
    with gr.Row():
        mujoco_output_image = gr.Image(type="numpy", label="MuJoCo Rendered Scene")
        gaussian_output_image = gr.Image(type="numpy", label="Gaussian Rendered Scene")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### MuJoCo Parameters")
            mujoco_joint_inputs = [gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label=f"MuJoCo θ{i+1}", show_label=True) for i in range(n_joints)]
        with gr.Column():
            gr.Markdown("### Gaussian Parameters (Optimized)")
            gaussian_joint_inputs = [gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label=f"Gaussian θ{i+1}", show_label=True, interactive=False) for i in range(n_joints)]
    
    with gr.Row():
        reset_demo_button = gr.Button("Reset Demo")
        reset_gaussians_button = gr.Button("Reset Gaussians to Robot")
    
    # Use gr.State to store the current parameters
    mujoco_params_state = gr.State([0] * n_joints)
    gaussian_params_state = gr.State([0] * n_joints)

    def update_and_render(new_value, current_mujoco_params, current_gaussian_params, param_index):
        current_mujoco_params[param_index] = new_value
        mujoco_image = render_scene(*current_mujoco_params[:n_joints])
        
        new_input_received.set()
        
        # Use current_gaussian_params as the starting point for optimization
        optimization_queue.put((current_mujoco_params[:n_joints], current_gaussian_params[:n_joints]))
        
        return mujoco_image, current_mujoco_params, current_gaussian_params

    # Add change event to MuJoCo joint inputs
    for i, input_component in enumerate(mujoco_joint_inputs):
        input_component.change(
            fn=update_and_render,
            inputs=[input_component, mujoco_params_state, gaussian_params_state, gr.State(value=i)],
            outputs=[mujoco_output_image, mujoco_params_state, gaussian_params_state],
            show_progress=False
        )
    
    def reset_all_params():
        reset_values = reset_params()
        return reset_values + reset_values + [reset_values, reset_values]

    def reset_demo():
        reset_values = reset_params()
        return reset_values + reset_values + [reset_values, reset_values]

    def reset_gaussians_to_robot(current_mujoco_params, current_gaussian_params):
        new_input_received.set()
        reset_to_mujoco.set()  # Set the flag to use MuJoCo params directly
        # Set the Gaussian parameters equal to the MuJoCo parameters
        current_gaussian_params = current_mujoco_params.copy()
        optimization_queue.put((current_mujoco_params, current_gaussian_params))
        # Return the updated Gaussian parameters and both states
        return current_mujoco_params + [current_mujoco_params, current_gaussian_params]

    reset_demo_button.click(
        fn=reset_demo,
        outputs=mujoco_joint_inputs + gaussian_joint_inputs + [mujoco_params_state, gaussian_params_state]
    )

    reset_gaussians_button.click(
        fn=reset_gaussians_to_robot,
        inputs=[mujoco_params_state, gaussian_params_state],
        outputs=gaussian_joint_inputs + [mujoco_params_state, gaussian_params_state]
    )
    
    # Start the optimization thread
    optimization_thread = start_optimization_thread()

    # Use a generator to stream optimization results
    def stream_optimization_results():
        for result, optimized_params in continuous_optimization():
            yield (result,) + tuple(optimized_params)

    demo.load(fn=initial_render, outputs=[mujoco_output_image, gaussian_output_image])
    demo.load(fn=stream_optimization_results, outputs=[gaussian_output_image] + gaussian_joint_inputs)

if __name__ == "__main__":
    try:
        demo.queue().launch(share=True, server_port=8900)
    finally:
        stop_optimization.set()
        optimization_thread.join()
        print("Optimization thread joined")
