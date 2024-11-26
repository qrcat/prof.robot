from collision.chain_utils import build_chain_relation_map
from collision.log import RegistorLog
from collision.dataset import DrRobotData
from collision.network import HyperNetwork
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Any, List, Tuple, Union

import pathlib
import argparse


class Config:
    def __init__(self, args):
        self.args = args

        self.ndf_activator = lambda x: torch.abs(x)

        self.ndf_threshold = 1e-5

    @property
    def get_metrics(self):
        if self.args.ndf:
            return nn.L1Loss
        else:
            return nn.BCEWithLogitsLoss

    def get_ndf(self, g: torch.Tensor, s: torch.Tensor):
        return self.ndf_activator(g)


def main(args):
    config = Config(args)

    @torch.no_grad()
    def eval_testset():
        metrics_eval = config.get_metrics(reduction="none")

        loss_count = []
        correct = 0
        total = 0
        for joint_positions, labels in ds_test:
            g, s = model(joint_positions)

            if args.ndf:
                f = config.get_ndf(g, s)
                loss_count.append(metrics_eval(f, labels))

                predicted = (f > config.ndf_threshold).float()
                labels = (labels > 0).float()
            else:
                f = g * s
                loss_count.append(metrics_eval(f, labels))

                predicted = (g > 0.0).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total, torch.cat(loss_count).mean()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)

    dataset = DrRobotData(
        data_path, t="distance" if args.ndf else "label"
    )  # for NDF, we precompute the distance
    ds_train, ds_test = dataset.get_split(batchsize=args.batchsize)

    scene_xml = data_path / args.scene_xml
    print("Loading scene xml file from {}", scene_xml.as_posix())
    relation_map, chain = build_chain_relation_map(scene_xml.as_posix())

    model = HyperNetwork(
        chain.n_joints if args.num_joints < 1 else args.num_joints, relation_map, flatten=args.flatten,
    )
    model.cuda()

    # lr=args.learning_rate

    optim = torch.optim.Adam(
        [
            {
                "params": model.encoder.parameters(),
            },
            {
                "params": model._mlp.parameters(),
            },
            {
                "params": model.s,
            },
        ],
        lr=args.learning_rate,
    )
    scler = torch.optim.lr_scheduler.MultiStepLR(
        optim, np.cumsum([args.epochs // 100 * i**2 for i in range(1, 7)]), 0.5
    )

    writer = SummaryWriter(output_path / "collision")

    tbar = trange(args.epochs)

    metrics = config.get_metrics()

    logger = RegistorLog(tbar, writer)

    data_loader = iter(ds_train)
    for epoch in tbar:
        if epoch % (10 * len(ds_train)) == 0:
            accuracy, eval_value = eval_testset()

            if args.ndf:
                # update NDF threshold
                config.ndf_threshold = eval_value.item() * 0.5
            else:
                ...

            logger.log_test(
                {
                    "accuracy": format(accuracy, ".4f"),
                    "L1" if args.ndf else "BCE": format(eval_value.item(), ".4f"),
                },
                epoch,
            )

            ds_train.shuffle()

        try:
            joint_positions, labels = next(data_loader)
        except StopIteration:
            data_loader = iter(ds_train)
            joint_positions, labels = next(data_loader)

        x = joint_positions.requires_grad_(True)
        dist, s = model(x)
        if args.ndf:
            dist = config.get_ndf(dist, s)

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

        if args.ndf:
            Ll1 = metrics(dist, labels)
            Leikonal = ((grad_val[labels[:, 0] > 0].norm(2, dim=-1) - 1) ** 2).mean()

            loss = Ll1 + 0.1 * Leikonal
        else:
            Lbce = metrics(dist * s, labels)
            Leikonal = ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()

            loss = Lbce if args.without_eikonal else Lbce + 0.1 * Leikonal

        optim.zero_grad()
        loss.backward()
        optim.step()
        scler.step()

        logger.log_train(
            {
                "L1" if args.ndf else "BCE": format(
                    Ll1.item() if args.ndf else Lbce.item(), ".4f"
                ),
                "eikonal": format(Leikonal.item(), ".4f"),
                "s": format(s.item(), ".4f"),
                "LR": scler.get_last_lr()[0],
            },
            epoch,
        )

    model_name = ""
    if args.ndf:
        model_name += "ndf_"
    else:
        model_name += "sdf_"
    if args.without_eikonal:
        model_name += "wo_eik_"
    if args.flatten:
        model_name += "flatten_"
    model_name += "net.ckpt"
    torch.save(model, output_path / model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/universal_robots_ur5e_collision"
    )
    parser.add_argument(
        "--output_path", type=str, default="output/universal_robots_ur5e_collision"
    )
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_joints", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batchsize", type=int, default=100000)
    parser.add_argument("--without_eikonal", action="store_true")
    parser.add_argument("--ndf", action="store_true")
    parser.add_argument("--flatten", action="store_true")

    args = parser.parse_args(
        [
            "--epochs",
            "100000",
            "--data_path",
            "data/universal_robots_ur5e_collision",
            "--output_path",
            "output/test",
        ]
    )

    main(args)
