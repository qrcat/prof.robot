#!/usr/bin/env python3
# refactor_train.py
"""
Refactored training script for collision SDF/NDF/Lipschitz networks.
- clearer device handling
- Trainer class with train / eval / save
- safer BCE/AUC handling
- fixes: avoid overwriting mujoco model variable, consistent use of s*outputs in metrics,
  guarded torch.jit.trace, scheduler stepping and logging
"""

from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import mujoco

from torch.utils.tensorboard import SummaryWriter

from typing import Any, Dict, Optional, Tuple

import pathlib
import argparse
from sklearn.metrics import roc_auc_score, roc_curve

# your project imports (keep same paths)
from utils.pk_utils import build_chain_from_mjcf_path
from utils.collision_utils import DrDynamicRobotData
from utils.collision_network import HyperEnbedding, HyperDynamicNetwork, LipMLP
from utils.steik_utils import StEikLoss


class MultiLogger:
    """Aggregate logger: tqdm progress bar + TensorBoard writer (optional)."""
    def __init__(self, pbar: Optional[tqdm] = None, tb_writer: Optional[SummaryWriter] = None):
        self.pbar = pbar
        self.tb_writer = tb_writer

    def log(self, metrics: Dict[str, float], step: int, phase: str = "train"):
        if self.pbar:
            # choice: set postfix in training, description on test
            if phase == "train":
                self.pbar.set_postfix(metrics)
            else:
                desc = ' '.join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                self.pbar.set_description(desc)
        if self.tb_writer:
            for k, v in metrics.items():
                # ensure scalar float
                self.tb_writer.add_scalar(f"{phase}/{k}", float(v), step)


def safe_to_device(tensor: torch.Tensor, device: torch.device):
    if tensor is None:
        return None
    return tensor.to(device)


def compute_binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """
    logits: tensor of shape (N,) floats (not sigmoid-ed)
    labels: tensor of shape (N,) floats (0/1)
    """
    result: Dict[str, Any] = {}
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        TP = int(((preds == 1) & (labels == 1)).sum().item())
        TN = int(((preds == 0) & (labels == 0)).sum().item())
        FP = int(((preds == 1) & (labels == 0)).sum().item())
        FN = int(((preds == 0) & (labels == 1)).sum().item())

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # AUC: guard against single-class label arrays which make roc_auc_score crash.
        try:
            auc = float(roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()))
        except Exception:
            auc = float("nan")

        # BCE loss scalar
        bce_loss = float(nn.BCEWithLogitsLoss()(logits, labels).item())

        # Compute ROC curve data for visualization
        roc_data = {}
        try:
            fpr, tpr, _ = roc_curve(labels.cpu().numpy(), probs.cpu().numpy())
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
        except Exception:
            roc_data = {"fpr": [], "tpr": [], "auc": auc}

        result.update({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "bce_loss": bce_loss,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "roc_data": roc_data
        })
    return result


class Trainer:
    def __init__(self, args: argparse.Namespace, device: Optional[torch.device] = None):
        self.args = args
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # paths
        self.data_path = pathlib.Path(args.data_path)
        self.output_path = pathlib.Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # dataset
        self.dataset = DrDynamicRobotData(self.data_path, t='label')
        self.ds_train, self.ds_test = self.dataset.get_split(batchsize=args.batchsize)

        # mujoco model (physical model) -> do not overwrite later
        scene_xml = self.data_path / args.scene_xml
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
        self.chain = build_chain_from_mjcf_path(scene_xml.as_posix(), args.root_name)
        parents = self.chain.get_joint_parent_frame_names()
        root_frame = self.chain.find_frame(args.root_name)
        self.remap_index = HyperEnbedding.build_remap_index(self.mj_model, self.chain)

        # create network (net) and move to device
        if args.model == 'lipschitz':
            net = LipMLP([self.mj_model.njnt, self.mj_model.njnt * 8, 512, 512, 512, 512, 1], no_scalar_s=args.no_scalar_s)
        else:
            net = HyperDynamicNetwork(self.mj_model.njnt, parents, self.remap_index, root_frame, flatten=args.flatten, no_scalar_s=args.no_scalar_s)

        self.net = net.to(self.device)

        # optimizer + scheduler
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        # build integer milestones for MultiStepLR
        milestones = list(map(int, np.cumsum([args.epochs // 100 * i ** 2 for i in range(1, 7)])))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=1e-6,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )


        # logging
        self.writer = SummaryWriter(self.output_path / 'dfnet_tb')
        self.tbar = trange(args.epochs, desc="Epoch")
        self.logger = MultiLogger(self.tbar, self.writer)

        # losses and options
        self.metrics_bce = nn.BCEWithLogitsLoss()
        self.steik = StEikLoss() if args.grad == "steik" else None

        # training state
        self.eval_time = args.eval_time if args.eval_time is not None else len(self.ds_train)
        # data iterator
        self.data_loader = iter(self.ds_train)
    @torch.no_grad()
    def eval_testset_quick(self) -> Tuple[float, float]:
        """A fast eval that returns (accuracy, mean BCE) similar to original behavior."""
        self.net.eval()
        bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for joint_positions, labels, pointclouds in self.ds_test:
                joint_positions = joint_positions.to(self.device)
                labels = labels.to(self.device).float()
                outputs, s = self.net(joint_positions, pointclouds)
                logits = outputs * s
                loss_per_sample = bce_loss_fn(logits, labels)
                losses.append(loss_per_sample.detach().cpu())
                preds = (logits.sigmoid() > 0.5).float()
                total += labels.size(0)
                correct += int((preds == labels).sum().item())
        mean_bce = float(torch.cat(losses).mean().item()) if len(losses) > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        self.net.train()
        return acc, mean_bce

    def comprehensive_evaluation(self) -> Dict[str, Any]:
        """Full evaluation across ds_test, returns metrics dict."""
        self.net.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for joint_positions, labels, pointclouds in self.ds_test:
                joint_positions = joint_positions.to(self.device)
                labels = labels.to(self.device).float()
                outputs, s = self.net(joint_positions, pointclouds)
                logits = outputs * s
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if len(all_logits) == 0:
            return {}

        logits = torch.cat(all_logits).view(-1)
        labels = torch.cat(all_labels).view(-1)
        metrics = compute_binary_metrics_from_logits(logits, labels)
        self.net.train()
        return metrics

    def train(self):
        self.net.train()
        for epoch in self.tbar:
            # periodic eval
            if (epoch % self.eval_time == 0) or (epoch == self.args.epochs - 1):
                acc, bce_eval = self.eval_testset_quick()
                self.logger.log({"accuracy": acc, "BCE": bce_eval}, epoch, "test")
                # if ds_train supports shuffle (original did), call it
                if hasattr(self.ds_train, "shuffle"):
                    try:
                        self.ds_train.shuffle()
                    except Exception:
                        pass

            # fetch batch
            try:
                joint_positions, labels, pointclouds = next(self.data_loader)
            except StopIteration:
                self.data_loader = iter(self.ds_train)
                joint_positions, labels, pointclouds = next(self.data_loader)

            # prepare tensors on device
            joint_positions = joint_positions.to(self.device)
            labels = labels.to(self.device).float()
            pointclouds = pointclouds.to(self.device).float()

            loss_log = {}

            if self.args.model == 'lipschitz':
                dist, s = self.net(joint_positions)
                # net specific lipschitz penalty
                if hasattr(self.net, "get_lipschitz_loss"):
                    Llip = self.net.get_lipschitz_loss()
                else:
                    Llip = torch.tensor(0.0, device=self.device)
                loss_log['lip'] = float(Llip.item()) if isinstance(Llip, torch.Tensor) else float(Llip)

                logits = dist  # in lipschitz net dist probably is already logits
            else:
                # non-lipschitz net: compute dist (logits) and optionally eikonal/steik
                x = joint_positions.requires_grad_(True)
                dist, s = self.net(x, pointclouds)
                logits = dist * s  # consistent with evaluation
                    
                if self.args.grad == "none":
                    Leikonal = torch.tensor(0.0, device=self.device)
                elif self.args.grad == "base":
                    # compute gradient for eikonal term
                    d_points = torch.ones_like(dist, device=self.device)
                    grad_val = torch.autograd.grad(
                        outputs=dist,
                        inputs=x,
                        grad_outputs=d_points,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                        allow_unused=True
                    )[0]  # can be None if allow_unused True
                    if grad_val is None:
                        Leikonal = torch.tensor(0.0, device=self.device)
                    else:
                        if self.args.flatten:
                            Leikonal = ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
                        else:
                            # safety: use remap_index if exists
                            idx = getattr(self.net.encoder, "remap_index", self.remap_index)
                            Leikonal = ((grad_val[:, idx].norm(2, dim=-1) - 1) ** 2).mean()
                    loss_log['Leikonal'] = float(Leikonal.item())
                elif self.args.grad == "steik" and self.steik is not None:
                    steik_loss = self.steik(joint_positions, dist)
                    Leikonal = steik_loss.get('eikonal_term', torch.tensor(0.0, device=self.device))
                    loss_log['Linter'] = float(steik_loss.get('inter_term', 0.0))
                    loss_log['Ldiv'] = float(steik_loss.get('div_loss', 0.0))


            # BCE
            Lbce = self.metrics_bce(logits, labels)
            loss_log["BCE"] = float(Lbce.item())

            # total loss composition
            if self.args.model == 'lipschitz':
                loss = Lbce + 0.1 * Llip
            else:
                if self.args.grad == "steik" and self.steik is not None:
                    loss = Lbce + 0.001 * steik_loss['loss']
                    # optionally update divergence weight in steik (original behavior)
                    if hasattr(self.steik, "update_div_weight"):
                        self.steik.update_div_weight(epoch, self.args.epochs, [0.0, 0.5, 0.75, 1.0])
                else:
                    loss = Lbce + 0.1 * (Leikonal if 'Leikonal' in locals() else torch.tensor(0.0, device=self.device))

            # learning rate for logging
            try:
                current_lr = self.scheduler.get_last_lr()[0]
            except Exception:
                current_lr = self.optim.param_groups[0]['lr']

            loss_log['s'] = float(s.item()) if isinstance(s, torch.Tensor) else float(s)
            loss_log['lr'] = float(current_lr)

            # backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Scheduler step per iteration (original used step after each iter)
            try:
                self.scheduler.step()
            except Exception:
                pass

            # log
            self.logger.log(loss_log, epoch, "train")

        # end for epoch
        # final evaluation and save
        eval_results = self.comprehensive_evaluation()
        self._save_eval_results(eval_results)
        self._save_model_trace()

    def _save_eval_results(self, metrics: Dict[str, Any]):
        # print and write to file
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and (not np.isnan(v)) else str(v)

        print("Final evaluation results:")
        for k, v in metrics.items():
            if k in ("TP", "TN", "FP", "FN"):
                print(f"{k}: {v}")

        eval_file = self.output_path / "evaluation_results.txt"
        with open(eval_file, 'w') as f:
            for k, v in metrics.items():
                if k != "roc_data":  # Skip roc_data for text file
                    f.write(f"{k}: {v}\n")
        print(f"Saved evaluation -> {eval_file}")
        
        # Save ROC curve data to separate JSON file
        if "roc_data" in metrics:
            import json
            roc_file = self.output_path / f"roc_data_{self.args.model}_{self.args.grad}_{'flatten' if self.args.flatten else 'layered'}.json"
            with open(roc_file, 'w') as f:
                json.dump(metrics["roc_data"], f, indent=2)
            print(f"Saved ROC data -> {roc_file}")

    def _save_model_trace(self):
        # safe trace: many custom modules might not be traceable; do in try/except
        trace_name = f"{self.args.model}_" + (self.args.grad if self.args.model == "sdf" else "") + ("flatten_" if self.args.flatten else "") + "net.pt"
        trace_path = self.output_path / trace_name
        try:
            # try to find a reasonable num_joints attribute
            example_dim = getattr(self.net, "num_joints", None)
            if example_dim is None and hasattr(self.net, "encoder"):
                example_dim = getattr(self.net.encoder, "num_joints", None) or getattr(self.net.encoder, "remap_index", None)
                if isinstance(example_dim, (list, tuple, np.ndarray)):
                    example_dim = int(len(example_dim))
            if example_dim is None:
                example_input = torch.zeros(1, self.mj_model.njnt), torch.zeros(1, 100, 3)
            else:
                example_input = torch.zeros(1, int(example_dim)), torch.zeros(1, 100, 3)
            
            net_cpu = self.net.to('cpu').eval()
            try:
                traced = torch.jit.trace(net_cpu, example_input)
                traced.save(trace_path)
                print(f"Saved torch.jit.trace -> {trace_path}")
            finally:
                # move back to original device
                self.net.to(self.device)
        except Exception as e:
            print(f"Warning: failed to torch.jit.trace model: {e}. Skipping trace save.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/universal_robots_ur5e_dynamic_collision")
    parser.add_argument("--output_path", type=str, default="output/universal_robots_ur5e_dynamic_collision")
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--root_name", type=str, default="base")
    parser.add_argument("--batchsize", type=int, default=5000)
    parser.add_argument("--eval_time", type=int)
    parser.add_argument("--without_eikonal", action="store_true")
    parser.add_argument("--model", choices=['sdf', 'ndf', 'lipschitz'], default='sdf')
    parser.add_argument("--grad", choices=["base", "none", "steik"], default='base')
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--no-scalar-s", action="store_true")
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs"
    )
    return parser.parse_args()
    


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
