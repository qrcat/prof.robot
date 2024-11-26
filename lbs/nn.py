import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import open3d as o3d
from utils.lbs_utils import PointCloudDataset
from tqdm import tqdm
# from pytorch3d.loss import chamfer_distance
import traceback
from lbs.lbs import lrs
from utils.chamferdist_utils import chamfer_distance, mean_chamfer_distance
# from torch.cuda.amp import autocast, GradScaler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

class IRS(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=3):
        super(IRS, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Forward pass through all but the last layer with GELU activation
        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        
        # No activation on the output layer
        x = self.layers[-1](x)
        return x

def train_lrs(gaussians, implicit=True):

    opt = gaussians.args   
    pc_path = f"{opt.dataset_path}/points3D_multipleview.ply"
    pc_pcd = o3d.io.read_point_cloud(pc_path)
    pc_np = np.asarray(pc_pcd.points)
    v_template = torch.tensor(pc_np, dtype=torch.float32, device='cuda')

    # v_template = torch.tensor(gaussians.get_xyz, dtype=torch.float32)
    # breakpoint()

    # save v_template as ply
    # v_template_pcd = o3d.geometry.PointCloud()
    # v_template_pcd.points = o3d.utility.Vector3dVector(v_template.cpu().numpy())
    # o3d.io.write_point_cloud('v_template.ply', v_template_pcd)

    index = 0
   
    path = opt.dataset_path
    sample_path = os.path.join(path, f'sample_{index}')

    gaussians.training_setup(opt)

    # chain = pk.build_chain_from_mjcf("mujoco_menagerie/shadow_hand/scene_left.xml")
    V = v_template.shape[0]
    J = gaussians.num_joints
    N = 8

    print("[Train] LRS dataloader")
    dataset = PointCloudDataset(path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    v_template = v_template[None].repeat(N, 1, 1).cuda()

    compiled_chamfer_distance = torch.compile(mean_chamfer_distance, fullgraph=True)
    scaler = GradScaler()
    

    for epoch in tqdm(range(opt.lrs_train_epochs)):
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}')
        running_average = 0.0
        for i, (pcd, pose) in enumerate(dataloader):
            try:
                pcd = pcd.cuda()
                pose = pose.cuda()
                
                gaussians.optimizer_lrs.zero_grad()
                
                with autocast('cuda'):
                    v_deformed, _ = lrs(pose, 
                                        v_template, 
                                        None, 
                                        gaussians.chain, 
                                        lrs_model = gaussians.lrs_model if implicit else None)
                    loss = compiled_chamfer_distance(pcd, v_deformed)

                scaler.scale(loss).backward()
                if i == 0:
                    running_average = loss.item()
                running_average = 0.99 * running_average + 0.01 * loss.item()

                if running_average < 1e-4:
                    pbar.write("Loss is less than 1e-4, breaking")
                    break

                scaler.step(gaussians.optimizer_lrs)
                scaler.update()

                pbar.set_description(f"Loss: {loss.item():.7f}, Running Average: {running_average:.7f}")
                pbar.update(1)
            except Exception as e:
                print(e)
                traceback.print_exc()
                print("oof, something went wrong")
            if i == opt.lrs_train_steps:
                break
