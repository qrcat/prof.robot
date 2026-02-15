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
from utils.collision_utils import DrDynamicRobotData
from utils.collision_network import (
    HyperEnbedding,
    HyperNetwork,
    LipMLP,
    HyperDynamicNetwork,
)
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
                    loger.set_description(
                        " ".join(
                            [f"{key}:{item}" for key, item in ordered_dict.items()]
                        )
                    )
            elif isinstance(loger, SummaryWriter):
                for key, value in ordered_dict.items():
                    loger.add_scalar(f"{phase}/{key}", value, step)


def main(args):
    @torch.no_grad()
    def eval_testset():
        metrics_eval = nn.BCEWithLogitsLoss(reduction="none")
        model.eval()
        bce_count = []
        correct = 0
        total = 0
        for joint_positions, labels, pointclouds in ds_test:
            outputs, s = model(joint_positions, pointclouds)
            bce_count.append(metrics_eval(outputs * s, labels))
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model.train()
        
        return correct / total, torch.cat(bce_count).mean()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)

    dataset = DrDynamicRobotData(data_path, t="label")
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
        model = HyperDynamicNetwork(
            model.njnt, parents, remap_index, root_frame, init_s=2.3
        )
    model.cuda()

    if args.lipschitz:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.encoder.parameters(),
                },
                {
                    "params": model.pointnet.parameters(),
                },
                {
                    "params": model._mlp.parameters(),
                },
                {
                    "params": model.s,
                    "lr": 1e-3,
                },
            ],
            lr=args.learning_rate,
        )

    # Initialize warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    writer = SummaryWriter(output_path / "dfnet_tb")

    tbar = trange(args.epochs)

    metrics = nn.BCEWithLogitsLoss()

    logger = RegistorLog(tbar, writer)

    if args.eval_time is None:
        eval_time = len(ds_train)
    else:
        eval_time = args.eval_time

    if args.steik:
        steik = StEikLoss()

    data_loader = iter(ds_train)
    for epoch in tbar:
        if (epoch % eval_time == 0) or (epoch == args.epochs - 1):
            accuracy, bce_eval = eval_testset()
            logger.log(
                {
                    "accuracy": format(accuracy, ".4f"),
                    "BCE": format(bce_eval, ".4f"),
                },
                epoch,
                "test",
            )

            ds_train.shuffle()

        try:
            joint_positions, labels, pointclouds = next(data_loader)
        except StopIteration:
            data_loader = iter(ds_train)
            joint_positions, labels, pointclouds = next(data_loader)

        loss_dict = {}

        if args.lipschitz:
            dist, s = model(joint_positions, pointclouds)

            Llip = model.get_lipschitz_loss()

            loss_dict["lip"] = format(Llip.item(), ".4f")
        else:
            x = joint_positions.requires_grad_(True)
            dist, s = model(x, pointclouds)
            if args.steik:
                steik_loss = steik(joint_positions, dist, model.encoder.remap_index)

                Leikonal = steik_loss["eikonal_term"]

                loss_dict["Linter"] = format(steik_loss["inter_term"].item(), ".4f")
                loss_dict["Ldiv"] = format(steik_loss["div_loss"].item(), ".4f")
            else:
                dist, s = model(x, pointclouds)
                d_points = torch.ones_like(dist).requires_grad_(False)

                grad_val = torch.autograd.grad(
                    outputs=dist,
                    inputs=x,
                    grad_outputs=d_points,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True,
                )[0]

                Leikonal = (
                    (grad_val[:, model.encoder.remap_index].norm(2, dim=-1) - 1) ** 2
                ).mean()

            loss_dict["Leikonal"] = format(Leikonal.item(), ".4f")

        Lbce = metrics(dist * s, labels)

        loss_dict["BCE"] = format(Lbce.item(), ".4f")

        if args.lipschitz:
            loss = Lbce + 0.001 * Llip
        else:
            if args.steik:
                loss = Lbce + 0.1 * steik_loss["loss"]

                steik.update_div_weight(epoch, args.epochs, [0.0, 0.5, 0.75, 1.0])
            else:
                loss = Lbce + 0.001 * Leikonal

        loss_dict["s"] = format(s.item(), ".4f")

        # Get current learning rate for logging
        loss_dict["lr"] = scheduler.get_last_lr()[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.log(loss_dict, epoch, "train")
    model.eval()
    output_name = "".join(["df_net", "_lipschitz" if args.lipschitz else "", ".pt"])
    trace = torch.jit.trace(
        model.cpu(),
        (torch.zeros(1, model.num_joints), torch.zeros(1, 100, 3)),
    )
    trace.save(output_path / output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/universal_robots_ur5e_dynamic_collision"
    )
    parser.add_argument(
        "--output_path", type=str, default="output/universal_robots_ur5e_collision"
    )
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--root_name", type=str, default="base")
    parser.add_argument("--batchsize", type=int, default=10000)
    parser.add_argument("--eval_time", type=int)
    parser.add_argument("--lipschitz", action="store_true")
    parser.add_argument("--steik", action="store_true")
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs"
    )

    args = parser.parse_args()

    main(args)
