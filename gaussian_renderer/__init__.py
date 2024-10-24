#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from lbs.lbs import lrs, pose_conditioned_deform



def render(viewpoint_camera, 
           pc : GaussianModel, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           stage='pose_conditioned'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    assert stage in ['canonical', 'pose_conditioned']


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )


    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

   
    joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    if stage == 'pose_conditioned':
        if pc.args.k_plane: # using k-plane deformation model for deformation instead of linear blend skinning
            means3D, scales, rotations, opacity, shs = pc._deformation(means3D, scales, rotations, opacity, shs, times_sel=None, joints=joints)
            rotations = pc.rotation_activation(rotations)
        else: # use lbs for means+rotations, + MLP for appearance
            means3D_deformed, rotations_deformed = lrs(joints[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=True, lrs_model=pc.lrs_model, rotations=rotations)
            
            if pc.args.no_appearance_deformation: # without learnable deformation model on scales, opacity, and spherical harmonics
                scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
            else:
                scales_out, _rotations_out, opacity_out, shs_out = \
                    pose_conditioned_deform(means3D[None], means3D_deformed[None], scales[None], rotations[None], \
                                            opacity[None], shs[None], joints[None].float(), pc.appearance_deformation_model)
            
            #these are calculated using joint transformations
            means3D = means3D_deformed[0]
            rotations = rotations_deformed[0]
            
            #these are learned during lrs
            scales = scales_out[0]
            opacity = opacity_out[0]
            shs = shs_out[0]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # viewmat = viewmat.requires_grad_(True)
    # print("viewmat shape", viewmat.shape, "supposed to be 4x4")
    # print("grad before", viewpoint_camera.v.grad)
    # breakpoint()

    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+D',
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0, :, :, :3].permute(2, 0, 1)
    rendered_depth = render_colors[0, :, :, 3] # depth map
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii}
