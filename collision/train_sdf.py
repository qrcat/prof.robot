from chain_utils import build_chain_relation_map


from dataset import DrRobotData
from network import HyperNetwork
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Any, List, Tuple, Union

import pathlib
import argparse


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
    relation_map, chain = build_chain_relation_map(scene_xml.as_posix())

    model = HyperNetwork(chain.n_joints, relation_map)
    model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scler = torch.optim.lr_scheduler.MultiStepLR(optim, np.cumsum([args.epochs//100*i**2 for i in range(1, 7)]), 0.5)

    writer = SummaryWriter(output_path / 'dfnet_tb')

    tbar = trange(args.epochs)

    metrics = nn.BCEWithLogitsLoss()

    logger = RegistorLog(tbar, writer)

    data_loader = iter(ds_train)
    for epoch in tbar:
        if epoch % (10 * len(ds_train)) == 0:
            accuracy, bce_eval = eval_testset()
            logger.log({
                "accuracy": format(accuracy, ".4f"),
                "BCE": format(bce_eval, ".4f"),
            }, epoch, "test")
            
            ds_train.shuffle()

        ds_train
        try:
            joint_positions, labels = next(data_loader)
        except StopIteration:
            data_loader = iter(ds_train)
            joint_positions, labels = next(data_loader)

        x = joint_positions.requires_grad_(True)
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
        
        Lbce = metrics(dist*s, labels)
        Leikonal = ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()

        loss = Lbce + 0.1 * Leikonal

        optim.zero_grad()
        loss.backward()
        optim.step()
        scler.step()

        logger.log({
            "BCE": format(loss.item(), ".4f"),
            "eikonal": format(Leikonal.item(), ".4f"),
            "s": format(s.item(), ".4f"),
            "LR": scler.get_last_lr()[0],
        }, epoch, "train")
    

    torch.save(model.state_dict(), output_path / 'df_net.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/universal_robots_ur5e_collision")
    parser.add_argument("--output_path", type=str, default="output/universal_robots_ur5e_collision")
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batchsize", type=int, default=100000)

    args = parser.parse_args()

    main(args)
