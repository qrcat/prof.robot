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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import RobotScene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import itertools
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from lbs.nn import train_lrs


from torch.utils.data import DataLoader

def training(gaussians, scene, stage, tb_writer, dataset, opt, pipe, test_every, save_every, checkpoint_every, checkpoint, debug_from):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("bg color", background)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    max_iterations = opt.canonical_training_iterations if stage == 'canonical' else opt.pose_conditioned_training_iterations
    progress_bar = tqdm(range(first_iter, max_iterations), desc="Training progress")
    first_iter += 1
    # for iteration in range(first_iter, opt.iterations + 1):   

    viewpoint_stack = scene.getTrainCameras(stage=stage)
    iteration = 0

    for viewpoint_cam in itertools.cycle(viewpoint_stack):

        iteration += 1
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if opt.random_background:
            background = torch.rand_like(background)

        render_pkg = render(viewpoint_cam, gaussians, background, stage=stage)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        joint_pose = viewpoint_cam.joint_pose.float().requires_grad_(True)
        dist = gaussians._df_net(joint_pose)
        d_points = torch.ones_like(dist).requires_grad_(False)
        # TODO: add eikonal loss here
        grad_val = torch.autograd.grad(
            outputs=dist,
            inputs=joint_pose,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        eikonal_loss = ((grad_val.norm(2, dim=-1) - 1) ** 2)
        
        gaussians._df_net(viewpoint_cam.joint_pose.float())

        if opt.random_background:
            robot_mask = viewpoint_cam.robot_mask.cuda()
            gt_image = gt_image * robot_mask + background.reshape(-1, 1, 1) * ~robot_mask

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            background = torch.ones_like(background) if opt.random_background else torch.zeros_like(background)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), test_every, scene, render, background, stage)
            if (iteration % save_every == 0):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max memory used: {mem:.2f} GB")
                scene.save(iteration, stage)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                
            # CLIP gradient norm so weird camera angles can't mess this up
            for idx, param_group in enumerate(gaussians.optimizer.param_groups):
                grad_norm = torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=1.0)
                tb_writer.add_scalar(f"grad_norm_{idx}", grad_norm, iteration)
                
            if stage == 'pose_conditioned':
                grad_norm_lrs = torch.nn.utils.clip_grad_norm_(gaussians.lrs_model.parameters(), max_norm=1.0)
                tb_writer.add_scalar(f"grad_norm_lrs", grad_norm_lrs, iteration)
                grad_norm_appearance_deform = torch.nn.utils.clip_grad_norm_(gaussians.appearance_deformation_model.parameters(), max_norm=1.0)
                tb_writer.add_scalar(f"grad_norm_appearance_deform", grad_norm_appearance_deform, iteration)

            gaussians.optimizer.step()
            if stage == 'pose_conditioned':
                gaussians.optimizer_appearance_deformation.step()
                gaussians.optimizer_lrs.step()

            gaussians.optimizer.zero_grad()
            if stage == 'pose_conditioned':
                gaussians.optimizer_appearance_deformation.zero_grad()
                gaussians.optimizer_lrs.zero_grad()

            if (iteration % checkpoint_every == 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")

            assert stage in ['canonical', 'pose_conditioned']
            if iteration > max_iterations:
                break
      

def prepare_output_and_logger(args, experiment_name):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", experiment_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : RobotScene, renderFunc, background, stage):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    test_cameras = scene.getSampleCameras(stage=stage)
    if iteration % args.test_every == 0:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 
                               'cameras': test_cameras}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, background, stage=stage)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 16):
                        tb_writer.add_images(config['name'] + f"_view_{idx}/{stage}_render", image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_view_{idx}/{stage}_ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def three_stage_training(dataset, hyper, opt, pipe, test_every, save_every, checkpoint_iterations, checkpoint, debug_from, experiment_name):
    
    tb_writer = prepare_output_and_logger(dataset, experiment_name)
    gaussians = GaussianModel(dataset.sh_degree, opt)
    scene = RobotScene(dataset, gaussians, opt_params=opt)

    print("============ Training canonical Gaussian ============")
    # first stage, train the canonical gaussian
    training(gaussians, scene, 'canonical', tb_writer, dataset, opt, pipe, test_every, save_every, checkpoint_iterations, checkpoint, debug_from)

    print("============ Training LBS model ============")
    # second stage
    train_lrs(gaussians)

    print("============ Training pose conditioned Gaussian ============")
    #first stage, train the canonical gaussian
    training(gaussians, scene, 'pose_conditioned', tb_writer, dataset, opt, pipe, test_every, save_every, checkpoint_iterations, checkpoint, debug_from)


if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=randint(10000, 65535))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_every", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=4000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=4000)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--experiment_name", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    if args.experiment_name is None:
        import time 
        args.experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    three_stage_training(lp.extract(args), None, op.extract(args), pp.extract(args), args.test_every, args.save_every, args.checkpoint_every, args.start_checkpoint, args.debug_from, args.experiment_name)

    # All done
    print("\nTraining complete.")
