"""
How to use:
1. Train a model using train.py, it will create a new directory in output/
2. Run this script
python mujoco_app_realtime.py --model_path output/[path_to_your_model_directory]
"""


from itertools import cycle
from random import choice
import sys
import tempfile

import os

from matplotlib import pyplot as plt
from utils.mujoco_utils import get_canonical_pose
os.environ['MUJOCO_GL'] = 'egl'

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir
import clip
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
# background_color_list = [1, 1, 1]
background_color = background_color.float().cuda()
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
from torchvision.transforms import transforms as T


from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
for param in clip_model.parameters():
    param.requires_grad = False
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
for param in dinov2_vitb14.parameters():
    param.requires_grad = False
dinov2_vitb14.to(device)

class DummyCam:
    def __init__(self, azimuth, elevation, distance):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def gaussian_render_scene(*joint_angles):
    azimuth, elevation, distance = 0, -45, 1.0  # Fixed camera parameters
    
    dummy_cam = DummyCam(azimuth, elevation, distance)
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)

    joint_angles = torch.tensor(joint_angles)
    example_camera_mujoco = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 1.0, 1.0,\
                            224, 224, joint_pose=joint_angles, zero_init=True).cuda()
    frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
    return frame.detach().cpu().numpy().transpose(1, 2, 0)

def reset_params():
    new_input_received.set()
    return [0.0] * n_joints

def initial_render():
    initial_params = reset_params()
    print("initial_params: ", initial_params, "rendering scene...")
    # mujoco_image = render_scene(*initial_params)
    gaussian_image = gaussian_render_scene(*initial_params)
    print("done rendering scene")

    optimization_queue.put((None, "Victory Sign", initial_params, (0.02, "dot"), None))

    return gaussian_image

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
            print(params[0])
            if params is not None:
                image_input, text_input, gaussian_params, lr_info, camera_info = params
                
                lr, loss_mode = lr_info

                if image_input is not None:
                    print("Starting new optimization with image")
                    with torch.no_grad():
                        image_input_t = clip_preprocess(images=image_input, return_tensors="pt", padding=False)['pixel_values']
                        embedding_input = None
                        dino_features = dinov2_vitb14(image_input_t.cuda())
                else:
                    print("Starting new optimization with text:", text_input)
                    with torch.no_grad():
                        # text_input_t = clip.tokenize([text_input]).to(device)
                        # embedding_input = clip_model.encode_text(text_input_t)
                        inputs = clip_preprocess(text=[text_input], return_tensors="pt", padding=False)
                        for key in inputs.keys():
                            inputs[key] = inputs[key].to(device)

                        # text_embedding_output = clip_model.text_model(**inputs)

                        # text_embedding_output['last_hidden_state']
                        # embedding_input = text_embedding_output['pooler_output']

                        embedding_input = clip_model.get_text_features(**inputs)

                        dino_features = None
                    
                new_input_received.clear()  # Clear the flag before starting optimization
                
                # Use mujoco_params if reset_to_mujoco is set, otherwise use last_optimized_params or gaussian_params
                if reset_to_mujoco.is_set():
                    start_params = text_input
                    reset_to_mujoco.clear()  # Clear the flag after using it
                else:
                    start_params = last_optimized_params if last_optimized_params is not None else gaussian_params
                
                for loss_list, result, optimized_params in optimize(embedding_input, dino_features, gaussian_params, initial_lr=lr, loss_mode=loss_mode, camera_list=camera_info):
                    if new_input_received.is_set():
                        print("New input received, restarting optimization")
                        break
                    last_optimized_params = optimized_params  # Update last_optimized_params
                    
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        # print("yielding!")
                        plt.figure()
                        plt.plot(loss_list, label='Loss', marker='o')
                        plt.title('Loss Function')
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss Value')
                        plt.legend()
                        plt.grid()

                        yield plt, result, optimized_params
                        last_update_time = current_time
        except queue.Empty:
            print("Optimization queue is empty")
    print("Continuous optimization thread stopped")

def optimize(embedding_input, dino_features, gaussian_params, initial_lr=0.02, decay_factor=0.95, decay_steps=50, loss_mode="dot", camera_list=None):
    def nclamp(input, min, max):
        return input.clamp(min=min, max=max).detach() + input - input.detach()
    
    print("initial pose:", gaussian_params)
    
    joint_params = torch.nn.Parameter(
        torch.tensor(gaussian_params, dtype=torch.float32),
        requires_grad=True
    )
    joint_angles = nclamp(joint_params, -1, 1)

    if camera_list is None:
        # azimuth, elevation, distance = 0, -90, 0.3  # Fixed camera parameters
        camera_extr_params = (
            # ( 90, -90, 0.3, -0.15,  0.0 ),
            # (-90, -90, 0.3,  0.15,  0.0 ),
            # ( 0 ,  90, 0.3,  0.0 , -0.15),
            # ( 0 , -80, 0.3,  0.0 ,  0.15),
            # ( 0 , -85, 0.3,  0.0 ,  0.15),
            ( 0 , -90, 0.1, 0.05,  0.15),
        )
    else:
        camera_extr_params = (
            camera_list,
        )

    camera_list = []
    for azimuth, elevation, distance, pad_x, pad_y in camera_extr_params:
        dummy_cam = DummyCam(azimuth, elevation, distance)
        camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
        
        camera_extrinsic_matrix[0, 3] = pad_x
        camera_extrinsic_matrix[1, 3] = pad_y

        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), 1.4, 1.4, 480, 480, joint_pose=joint_angles, zero_init=True).cuda()

        camera_list.append(camera)

    optimizer = torch.optim.Adam([joint_params], lr=initial_lr)
    # optimizer = torch.optim.AdamW([joint_angles], lr=initial_lr, weight_decay=0.1)
    # optimizer = torch.optim.AdamW([joint_params], lr=initial_lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    # New variables for tracking optimization progress
    previous_loss = float('inf')
    stagnant_iterations = 0
    max_stagnant_iterations = 1000
    loss_threshold = 1e-3
    iteration = 0
    N = 2  # Yield every N iterations

    # T.Resize(224, T.InterpolationMode('bicubic')),
    preprocess = T.Compose([
        T.Resize(224),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    
    camera_loop = cycle(camera_list)

    loss_history = []


    while True:
        camera = next(camera_loop)

        # joint_params = joint_params + torch.randn_like(joint_params) * joint_width/100 * initial_lr

        joint_angles = nclamp(joint_params, -1, 1)
        camera.joint_pose = joint_angles

        gaussian_tensor = render(camera, gaussians, background_color)['render']

        gaussian_tensor_processed = preprocess(gaussian_tensor) 
        # gaussian_tensor_processed = gaussian_tensor

        if dino_features is not None:
            align_features = dino_features
            
            image_embedding = dinov2_vitb14(gaussian_tensor_processed[None])    
        else:
            align_features = embedding_input
            image_embedding = clip_model.get_image_features(gaussian_tensor_processed[None])

        if loss_mode == "cos":
            loss = 1-torch.cosine_similarity(image_embedding, align_features.detach(), dim=1)
        elif loss_mode == "l1":
            loss = torch.nn.functional.l1_loss(image_embedding, align_features.detach())
        elif loss_mode == "l2":
            loss = torch.nn.functional.mse_loss(image_embedding, align_features.detach())
        else:
            loss = -torch.matmul(image_embedding, align_features.T.detach())
        

        loss.backward()

        # if iteration % len(camera_list) == 0:
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        current_loss = loss.item()

        loss_history.append(loss.item())

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
            yield loss_history, torch.clamp(gaussian_tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy(), joint_angles.detach().cpu().numpy()

        # Stop optimization if loss has stagnated for too long
        if stagnant_iterations >= max_stagnant_iterations:
            # print(f"Optimization stopped: Loss stagnated for {max_stagnant_iterations} iterations")
            loss_history = []
            break

        if stop_optimization.is_set() or new_input_received.is_set():
            # Yield the last image before breaking
            loss_history = []
            yield loss_history, torch.clamp(gaussian_tensor.permute(1, 2, 0).detach().cpu(), 0, 1).numpy(), joint_angles.detach().cpu().numpy()
            break

    print("Optimization finished")

def start_optimization_thread():
    optimization_thread = threading.Thread(target=continuous_optimization)
    optimization_thread.start()
    return optimization_thread

with gr.Blocks() as demo:
    with gr.Row():
        control_image = gr.Image(type="pil", label="Control Image")
        gaussian_output_image = gr.Image(type="numpy", label="Gaussian Rendered Scene")

    with gr.Row():
        with gr.Column():

            control_text = gr.Textbox('A Robot do "Victory Sign"', label="Control Text")

            with gr.Row():
                control_loss = gr.Radio(
                    ["dot", "cos", "l1", "l2"],
                    value="dot",
                    type='value', 
                    label="loss"
                )
                control_lr = gr.Slider(minimum=0.0, maximum=1.0, value=0.02, step=0.001, label=f"LR", show_label=True)
            
            align_embedding_button = gr.Button("Align")
            reset_demo_button = gr.Button("Reset Gaussians")
            train_loss_output = gr.Plot()


            camera_azimuth = gr.Slider(minimum=-180, maximum=180, value=0, step=1, label=f"azimuth", show_label=True)
            camera_elevation = gr.Slider(minimum=-180, maximum=180, value=-90, step=1, label=f"elevation", show_label=True)
            camera_distance = gr.Slider(minimum=0, maximum=3, value=0.2, step=0.1, label=f"distance", show_label=True)
            camera_pad_x = gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label=f"elevation", show_label=True)
            camera_pad_y = gr.Slider(minimum=-1, maximum=1, value=0.15, step=0.01, label=f"elevation", show_label=True)
        with gr.Column():
            
            gr.Markdown("### Gaussian Parameters (Optimized)")
            gaussian_joint_inputs = [gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label=f"Gaussian θ{i+1}", show_label=True, interactive=False) for i in range(n_joints)]
    
    # Use gr.State to store the current parameters
    gaussian_params_state = gr.State([0] * n_joints)
    
    def reset_all_params():
        reset_values = reset_params()
        return reset_values + reset_values + [reset_values, reset_values]

    def reset_demo():
        reset_values = reset_params()
        return reset_values

    def align_text2image(clip_image_input, clip_text_input, current_gaussian_params, loss_lr, loss_mode, azimuth, elevation, distance, pad_x, pad_y):
        new_input_received.set()
        reset_to_mujoco.set()  # Set the flag to use MuJoCo params directly
        # Set the Gaussian parameters equal to the MuJoCo parameters
        loss_info = loss_lr, loss_mode
        camera_info = azimuth, elevation, distance, pad_x, pad_y
        optimization_queue.put((clip_image_input, clip_text_input, current_gaussian_params, loss_info, camera_info))
        # Return the updated Gaussian parameters and both states
        return current_gaussian_params
    
    align_embedding_button.click(
        fn=align_text2image,
        inputs=[control_image, control_text, gaussian_params_state, control_lr, control_loss, camera_azimuth, camera_elevation, camera_distance, camera_pad_x, camera_pad_y],
        outputs=gaussian_params_state
    )

    reset_demo_button.click(
        fn=reset_demo,
        outputs=gaussian_params_state
    )
    
    # Start the optimization thread
    optimization_thread = start_optimization_thread()

    # Use a generator to stream optimization results
    def stream_optimization_results():
        for loss_list, result, optimized_params in continuous_optimization():
            yield (loss_list, result,) + tuple(optimized_params)

    demo.load(fn=initial_render, outputs=[gaussian_output_image])
    demo.load(fn=stream_optimization_results, outputs=[train_loss_output, gaussian_output_image] + gaussian_joint_inputs)

if __name__ == "__main__":
    try:
        demo.queue().launch(share=True, server_port=8900)
    finally:
        stop_optimization.set()
        optimization_thread.join()
        print("Optimization thread joined")
