from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import mujoco

from torch.utils.tensorboard import SummaryWriter

from typing import Any, List, Tuple, Union

import pathlib
import argparse

from utils.pk_utils import build_chain_from_mjcf_path
from utils.collision_utils import DrRobotData
from utils.collision_network import HyperEnbedding, HyperNetwork, LipMLP
from utils.steik_utils import StEikLoss

class RegistorLog:
    def __init__(self, *args):
        self.logers = args

    def log(self, ordered_dict: dict, step: int, phase: str):
        for loger in self.logers:
            if isinstance(loger, tqdm):
                if phase == "train":
                    loger.set_postfix(ordered_dict)
                else:
                    loger.set_description(' '.join([f"{key}:{item}" for key, item in ordered_dict.items()]))
            elif isinstance(loger, SummaryWriter):
                for key, value in ordered_dict.items():
                    loger.add_scalar(f"{phase}/{key}", value, step)

def main(args):
    @torch.no_grad()
    def eval_testset():
        metrics_eval = nn.BCEWithLogitsLoss(reduction='none')

        bce_count = []
        correct = 0
        total = 0
        for joint_positions, labels in ds_test:
            outputs, s = model(joint_positions)
            bce_count.append(metrics_eval(outputs*s, labels))
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return correct / total, torch.cat(bce_count).mean()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)

    dataset = DrRobotData(data_path, t='label')
    ds_train, ds_test = dataset.get_split(batchsize=args.batchsize)

    scene_xml = data_path / args.scene_xml

    chain = build_chain_from_mjcf_path(scene_xml.as_posix(), args.root_name)

    model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
    parents = chain.get_joint_parent_frame_names()
    root_frame = chain.find_frame(args.root_name)
    remap_index = HyperEnbedding.build_remap_index(model, chain)

    if args.lipschitz:
        model = LipMLP([model.njnt, model.njnt * 8, 512, 512, 512, 512, 1])
    else:
        model = HyperNetwork(model.njnt, parents, remap_index, root_frame)
    model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scler = torch.optim.lr_scheduler.MultiStepLR(optim, np.cumsum([args.epochs//100*i**2 for i in range(1, 7)]), 0.5)

    writer = SummaryWriter(output_path / 'dfnet_tb')

    tbar = trange(args.epochs)

    metrics = nn.BCEWithLogitsLoss()

    logger = RegistorLog(tbar, writer)

    if args.eval_time is None:
        eval_time = 10 * len(ds_train)
    else:
        eval_time = args.eval_time

    if args.steki:
        steik = StEikLoss()

    data_loader = iter(ds_train)
    for epoch in tbar:
        if (epoch % eval_time == 0) or (epoch == args.epochs - 1):
            accuracy, bce_eval = eval_testset()
            logger.log({
                "accuracy": format(accuracy, ".4f"),
                "BCE": format(bce_eval, ".4f"),
            }, epoch, "test")
            
            ds_train.shuffle()

        try:
            joint_positions, labels = next(data_loader)
        except StopIteration:
            data_loader = iter(ds_train)
            joint_positions, labels = next(data_loader)

        loss_dict = {}

        if args.lipschitz:
            dist, s = model(joint_positions)

            Llip = model.get_lipschitz_loss()

            loss_dict['lip'] = format(Llip.item(), ".4f")
        else:
            x = joint_positions.requires_grad_(True)
            dist, s = model(x)
            if args.steki:
                steik_loss = steik(
                    joint_positions, dist, model.encoder.remap_index
                )

                Leikonal = steik_loss['eikonal_term']

                loss_dict['Linter'] = format(steik_loss['inter_term'].item(), ".4f")
                loss_dict['Ldiv'] = format(steik_loss['div_loss'].item(), ".4f")
            else:                
                dist, s = model(x)
                d_points = torch.ones_like(dist).requires_grad_(False)
                
                grad_val = torch.autograd.grad(
                    outputs=dist,
                    inputs=x,
                    grad_outputs=d_points,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
                
                Leikonal = ((grad_val[:, model.encoder.remap_index].norm(2, dim=-1) - 1) ** 2).mean()

            loss_dict['Leikonal'] = format(Leikonal.item(), ".4f")

        Lbce = metrics(dist*s, labels)
            
        loss_dict["BCE"] = format(Lbce.item(), ".4f")
        
        if args.lipschitz:
            loss = Lbce + 0.001 * Llip
        else:
            if args.steki:
                loss = Lbce + 0.1 * steik_loss['loss']

                steik.update_div_weight(epoch, args.epochs, [0.0, 0.5, 0.75, 1.0])
            else:
                loss = Lbce + 0.001 * Leikonal        

        loss_dict['s'] = format(s.item(), ".4f")
        loss_dict['lr'] = scler.get_last_lr()[0]

        optim.zero_grad()
        loss.backward()
        optim.step()
        scler.step()

        logger.log(loss_dict, epoch, "train")

    output_name =  ''.join(["df_net", "_lipschitz" if args.lipschitz else "", ".pt"])
    trace = torch.jit.trace(model.cpu(), torch.zeros(1, model.num_joints))
    trace.save(output_path / output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/universal_robots_ur5e_collision")
    parser.add_argument("--output_path", type=str, default="output/universal_robots_ur5e_collision")
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--root_name", type=str, default="base")
    parser.add_argument("--batchsize", type=int, default=100000)
    parser.add_argument("--eval_time", type=int)
    parser.add_argument("--lipschitz", action='store_true')
    parser.add_argument("--steki", action='store_true')

    args = parser.parse_args(
        [
            # '--data_path', 'data/xMate_SR3_collision',
            # '--output_path', 'output/xMate_SR3_collision',
            # '--root_name', 'xMateSR3_base',
            # '--lipschitz',
            # "--steki",

            '--data_path', 'data/universal_robots_ur5e_scene3_collision',
            '--output_path', 'output/universal_robots_ur5e_scene3_collision',
            '--root_name', 'base',
        ]
    )

    main(args)
