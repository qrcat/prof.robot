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
from utils.collision_utils import DrRobotData
from utils.collision_network import HyperEnbedding, HyperNetwork, LipMLP
from utils.steik_utils import StEikLoss


def otsu_threshold(distances: torch.Tensor) -> float:
    """
    Otsu's method
    Args:
        distances
    Returns:
        threshold
    """
    if len(distances) == 0:
        return 0.1
    
    dist_np = distances.cpu().numpy()
    
    hist, bin_edges = np.histogram(dist_np, bins=256)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    total_pixels = len(dist_np)
    
    hist_norm = hist.astype(float) / total_pixels
    
    max_var = 0
    optimal_threshold = 0.1
    
    for threshold_idx in range(1, len(hist_norm)):
        w0 = np.sum(hist_norm[:threshold_idx])
        w1 = np.sum(hist_norm[threshold_idx:])
        
        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.sum(bin_centers[:threshold_idx] * hist_norm[:threshold_idx]) / w0
        mean1 = np.sum(bin_centers[threshold_idx:] * hist_norm[threshold_idx:]) / w1
        
        between_var = w0 * w1 * (mean0 - mean1) ** 2
        
        if between_var > max_var:
            max_var = between_var
            optimal_threshold = bin_centers[threshold_idx]
    
    return float(optimal_threshold)


class MultiLogger:
    """Aggregate logger: tqdm progress bar + TensorBoard writer (optional)."""
    def __init__(self, pbar: Optional[tqdm] = None, tb_writer: Optional[SummaryWriter] = None):
        self.pbar = pbar
        self.tb_writer = tb_writer
        self.last_test_metrics = {}

    def log(self, metrics: Dict[str, float], step: int, phase: str = "train"):
        if self.pbar:
            if phase == "train":
                # self.pbar.set_postfix(metrics)
                pass
            else:
                # Test phase: display key test metrics
                self.last_test_metrics = metrics
                # Build test description
                if 'accuracy' in metrics:
                    acc = metrics['accuracy']
                    if 'auc' in metrics:
                        desc = f"Test | Acc: {acc:.3f} | AUC: {metrics['auc']:.3f}"
                    else:
                        desc = f"Test | Acc: {acc:.3f} | Loss: {metrics.get('bce_loss', metrics.get('mse_loss', 0)):.4f}"
                    
                    # Add other key metrics
                    if 'precision' in metrics and 'recall' in metrics:
                        desc += f" | P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f}"
                    
                    self.pbar.set_description(desc)
        
        if self.tb_writer:
            for k, v in metrics.items():
                # ensure scalar float
                self.tb_writer.add_scalar(f"{phase}/{k}", float(v), step)
    
    def get_test_summary(self) -> str:
        """Get summary information of test results"""
        if not self.last_test_metrics:
            return "No test results available"
        
        metrics = self.last_test_metrics
        summary = []
        
        if 'accuracy' in metrics:
            summary.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if 'auc' in metrics:
            summary.append(f"AUC: {metrics['auc']:.3f}")
        if 'precision' in metrics:
            summary.append(f"Precision: {metrics['precision']:.3f}")
        if 'recall' in metrics:
            summary.append(f"Recall: {metrics['recall']:.3f}")
        if 'f1_score' in metrics:
            summary.append(f"F1: {metrics['f1_score']:.3f}")
        
        return " | ".join(summary)


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
        self.dataset = DrRobotData(self.data_path, t='distance' if args.model == 'ndf' else 'label')
        self.ds_train, self.ds_test = self.dataset.get_split(batchsize=args.batchsize)

        # mujoco model (physical model) -> do not overwrite later
        scene_xml = self.data_path / args.scene_xml
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml.as_posix())
        self.chain = build_chain_from_mjcf_path(scene_xml.as_posix(), args.root_name)
        parents = self.chain.get_joint_parent_frame_names()
        root_frame = self.chain.find_frame(args.root_name)
        self.remap_index = HyperEnbedding.build_remap_index(self.mj_model, self.chain)

        init_s = 0.0 if args.no_scalar_s else 2.3
        # create network (net) and move to device
        if args.model == 'lipschitz':
            net = LipMLP([self.mj_model.njnt, self.mj_model.njnt * 8, 512, 512, 512, 512, 1])
        elif args.model == 'ndf':
            net = HyperNetwork(self.mj_model.njnt, parents, self.remap_index, root_frame, flatten=args.flatten, init_s=init_s)
        else:
            net = HyperNetwork(self.mj_model.njnt, parents, self.remap_index, root_frame, flatten=args.flatten, init_s=init_s)

        self.net = net.to(self.device)

        # optimizer + scheduler
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        # build integer milestones for MultiStepLR
        milestones = list(map(int, np.cumsum([args.epochs // 100 * i ** 2 for i in range(1, 7)])))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.5)

        # logging
        self.writer = SummaryWriter(self.output_path / 'dfnet_tb')
        self.tbar = trange(args.epochs, desc="Epoch")
        self.logger = MultiLogger(self.tbar, self.writer)

        # losses and options
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.steik = StEikLoss() if args.grad == "steik" else None

        # training state
        self.eval_time = args.eval_time if args.eval_time is not None else 10 * len(self.ds_train)
        # data iterator
        self.data_loader = iter(self.ds_train)

    @torch.no_grad()
    def eval_testset_quick(self) -> Dict[str, float]:
        """A fast eval that returns (accuracy, mean loss) similar to original behavior."""
        self.net.eval()
        
        losses = []
        correct = 0
        total = 0
        all_distances = []
        all_labels = []

        mse_loss = nn.MSELoss(reduction='none')
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        if self.args.model == 'ndf':
            for joint_positions, labels in self.ds_test:
                joint_positions, labels = joint_positions.to(self.device), labels.to(self.device).float()
                
                outputs, _ = self.net(joint_positions)

                dist = nn.functional.relu(outputs)
                # Save distances and labels for threshold calculation
                all_distances.append(dist.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                loss_per_sample = mse_loss(dist, labels)

                total += labels.size(0)

            if len(all_distances) > 0:
                distances = torch.cat(all_distances).view(-1)
                labels = torch.cat(all_labels).view(-1)
                
                # Automatically determine threshold using Otsu's method
                threshold = otsu_threshold(distances)
                
                preds = (distances > threshold).float()

                correct = int((preds == (labels > 0).float()).sum().item())
            else:
                correct = 0
        else:
            for joint_positions, labels in self.ds_test:
                joint_positions, labels = joint_positions.to(self.device), labels.to(self.device).float()
                
                outputs, s = self.net(joint_positions)
                
                logits = outputs * s
                loss_per_sample = bce_loss(logits, labels)
                preds = (logits.sigmoid() > 0.5).float()
                
                losses.append(loss_per_sample.detach().cpu())
                total += labels.size(0)
                
                correct += int((preds == labels).sum().item())
        
        mean_loss = float(torch.cat(losses).mean().item()) if len(losses) > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        self.net.train()

        return {
            "accuracy": acc,
            "mse_loss" if self.args.model == 'ndf' else 'bce_loss': mean_loss,
        }

    @torch.no_grad()
    def comprehensive_evaluation(self) -> Dict[str, Any]:
        """Full evaluation across ds_test, returns metrics dict."""
        self.net.eval()
        all_logits = []
        all_labels = []
        all_distances = []
        if self.args.model == 'ndf':
            for joint_positions, labels in self.ds_test:
                joint_positions, labels = joint_positions.to(self.device), labels.to(self.device).float()

                outputs, s = self.net(joint_positions)

                dist = nn.functional.relu(outputs)

                all_distances.append(dist.detach().cpu())
                all_labels.append(labels.detach().cpu())
            
            if len(all_distances) == 0:
                return {}
            
            distances = torch.cat(all_distances).view(-1)
            labels = torch.cat(all_labels).view(-1)
            labels = (labels > 0).float()
            
            # 使用大津法自动确定阈值
            threshold = otsu_threshold(distances)
            preds = (distances > threshold).float()
            
            TP = int(((preds == 1) & (labels == 1)).sum().item())
            TN = int(((preds == 0) & (labels == 0)).sum().item())
            FP = int(((preds == 1) & (labels == 0)).sum().item())
            FN = int(((preds == 0) & (labels == 1)).sum().item())

            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            mse_loss = float(nn.MSELoss()(distances, labels).item())
            
            # 为NDF模型计算ROC曲线：将不同的距离阈值视为不同的概率判断
            roc_data = self._compute_ndf_roc_curve(distances, labels)
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "mse_loss": mse_loss,
                "auc": roc_data['auc'],
                "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                "threshold": threshold,
                "otsu_threshold": threshold,
                "roc_data": roc_data
            }   
        else:
            for joint_positions, labels in self.ds_test:
                joint_positions, labels = joint_positions.to(self.device), labels.to(self.device).float()

                outputs, s = self.net(joint_positions)

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

    def _compute_ndf_roc_curve(self, distances: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """
        为NDF模型计算ROC曲线：使用对数距离然后套用sigmoid函数
        
        Args:
            distances: 距离值张量，形状 (N,)
            labels: 标签张量，形状 (N,)，值为0或1
            
        Returns:
            ROC曲线数据字典，包含fpr, tpr, auc等信息
        """
        with torch.no_grad():
            # 使用对数距离然后套用sigmoid函数
            # 为了避免对0取对数，添加一个小的epsilon
            epsilon = 1e-8
            log_distances = torch.log(distances + epsilon)
            
            probabilities = torch.sigmoid(log_distances)
            
            try:
                fpr, tpr, _ = roc_curve(labels.cpu().numpy(), probabilities.cpu().numpy())
                auc = float(roc_auc_score(labels.cpu().numpy(), probabilities.cpu().numpy()))
            except Exception:
                fpr, tpr, auc = [], [], float("nan")
            
            return {
                "fpr": fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
                "tpr": tpr.tolist() if isinstance(tpr, np.ndarray) else tpr,
                "auc": auc
            }

    def train(self):
        self.net.train()
        for epoch in self.tbar:
            # Regular evaluation
            if (epoch % self.eval_time == 0) or (epoch == self.args.epochs - 1):
                self.logger.log(self.eval_testset_quick(), epoch, "test")
                
                # If dataset supports shuffle, call it
                if hasattr(self.ds_train, "shuffle"):
                    try:
                        self.ds_train.shuffle()
                    except Exception:
                        pass

            # Get batch data
            try:
                joint_positions, labels = next(self.data_loader)
            except StopIteration:
                self.data_loader = iter(self.ds_train)
                joint_positions, labels = next(self.data_loader)

            # Prepare tensors on device
            joint_positions = joint_positions.to(self.device)
            labels = labels.to(self.device).float()

            loss_log = {}

            # Forward propagation and loss calculation based on model type
            if self.args.model == 'lipschitz':
                loss, loss_log = self._train_lipschitz(joint_positions, labels, loss_log, epoch)
            elif self.args.model == 'ndf':
                loss, loss_log = self._train_ndf(joint_positions, labels, loss_log)
            else:  # sdf model
                loss, loss_log = self._train_sdf(joint_positions, labels, loss_log, epoch)

            # Learning rate recording
            try:
                current_lr = self.scheduler.get_last_lr()[0]
            except Exception:
                current_lr = self.optim.param_groups[0]['lr']
            loss_log['lr'] = float(current_lr)

            # Backpropagation and optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Scheduler step
            try:
                self.scheduler.step()
            except Exception:
                pass

            # 更新进度条显示
            self._update_progress_bar(epoch, loss, loss_log, current_lr)

            # 记录到TensorBoard
            self.logger.log(loss_log, epoch, "train")

        # 最终评估和保存
        eval_results = self.comprehensive_evaluation()
        self._save_eval_results(eval_results)
        self._save_model_trace()
        
        # 显示最终训练总结
        print("\n" + "="*60)
        print(f"Training completed: {self.args.model.upper()} model")
        print(f"Final test results: {self.logger.get_test_summary()}")
        print("="*60)

    def _train_lipschitz(self, joint_positions, labels, loss_log, epoch):
        """训练Lipschitz模型"""
        dist, s = self.net(joint_positions)
        
        # Lipschitz惩罚
        if hasattr(self.net, "get_lipschitz_loss"):
            Llip = self.net.get_lipschitz_loss()
        else:
            Llip = torch.tensor(0.0, device=self.device)
        
        logits = dist
        Lbce = self.bce_loss(logits, labels)
        
        loss = Lbce + 0.1 * Llip
        
        loss_log.update({
            'lip': float(Llip.item()) if isinstance(Llip, torch.Tensor) else float(Llip),
            'BCE': float(Lbce.item()),
            's': float(s.item()) if isinstance(s, torch.Tensor) else float(s)
        })
        
        return loss, loss_log

    def _train_ndf(self, joint_positions, labels, loss_log):
        """训练NDF模型"""
        outputs, _ = self.net(joint_positions)
        dist = nn.functional.leaky_relu(outputs, negative_slope=0.001)
        
        loss = self.mse_loss(dist, labels)

        loss_log.update({
            "MSE": float(loss.item()),
        })
        
        return loss, loss_log

    def _train_sdf(self, joint_positions, labels, loss_log, epoch):
        """训练SDF模型"""
        x = joint_positions.requires_grad_(True)
        dist, s = self.net(x)
        logits = dist * s
        
        Lbce = self.bce_loss(logits, labels)
        
        # Eikonal损失计算
        if self.args.grad == "none":
            Leikonal = torch.tensor(0.0, device=self.device)
        elif self.args.grad == "base":
            d_points = torch.ones_like(dist, device=self.device)
            grad_val = torch.autograd.grad(
                outputs=dist,
                inputs=x,
                grad_outputs=d_points,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            if grad_val is None:
                Leikonal = torch.tensor(0.0, device=self.device)
            else:
                if self.args.flatten:
                    Leikonal = ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
                else:
                    idx = getattr(self.net.encoder, "remap_index", self.remap_index)
                    Leikonal = ((grad_val[:, idx].norm(2, dim=-1) - 1) ** 2).mean()
            loss_log['Leikonal'] = float(Leikonal.item())
        elif self.args.grad == "steik" and self.steik is not None:
            steik_loss = self.steik(joint_positions, dist)
            Leikonal = steik_loss.get('eikonal_term', torch.tensor(0.0, device=self.device))
            loss_log['Linter'] = float(steik_loss.get('inter_term', 0.0))
            loss_log['Ldiv'] = float(steik_loss.get('div_loss', 0.0))
            
            # 更新divergence权重
            if hasattr(self.steik, "update_div_weight"):
                self.steik.update_div_weight(epoch, self.args.epochs, [0.0, 0.5, 0.75, 1.0])
        
        # 总损失
        if self.args.grad == "steik" and self.steik is not None:
            loss = Lbce + 0.001 * steik_loss['loss']
        else:
            loss = Lbce + 0.1 * Leikonal
        
        loss_log.update({
            "BCE": float(Lbce.item()),
            's': float(s.item()) if isinstance(s, torch.Tensor) else float(s)
        })
        
        return loss, loss_log

    def _update_progress_bar(self, epoch, loss, loss_log, current_lr):
        """更新进度条显示"""
        # 构建显示字典
        if self.args.model == 'ndf':
            loss_display = {
                'loss': f"{loss.item():.4f}",
                'mse': f"{loss_log.get('MSE', 0):.4f}",
                'lr': f"{current_lr:.2e}",
            }
        elif self.args.model == 'lipschitz':
            loss_display = {
                'loss': f"{loss.item():.4f}",
                'bce': f"{loss_log.get('BCE', 0):.4f}",
                'lip': f"{loss_log.get('lip', 0):.4f}",
                'lr': f"{current_lr:.2e}",
                's': f"{loss_log.get('s', 0):.4f}"
            }
        else:  # sdf
            loss_display = {
                'loss': f"{loss.item():.4f}",
                'bce': f"{loss_log.get('BCE', 0):.4f}",
                'lr': f"{current_lr:.2e}",
                's': f"{loss_log.get('s', 0):.4f}"
            }
            
            # 添加eikonal损失显示
            if 'Leikonal' in loss_log:
                loss_display['eik'] = f"{loss_log['Leikonal']:.4f}"
            if 'Linter' in loss_log:
                loss_display['inter'] = f"{loss_log['Linter']:.4f}"
            if 'Ldiv' in loss_log:
                loss_display['div'] = f"{loss_log['Ldiv']:.4f}"

        # 更新描述
        model_name = self.args.model.upper()
        desc = f"{model_name} Epoch {epoch+1}/{self.args.epochs}"
        
        if self.args.model == 'ndf':
            desc += f" | Loss: {loss_display['loss']} | MSE: {loss_display['mse']}"
        else:
            desc += f" | Loss: {loss_display['loss']} | BCE: {loss_display['bce']}"
        
        desc += f" | LR: {loss_display['lr']}"
        
        # self.tbar.set_description(desc)
        self.tbar.set_postfix(loss_display)

    def _save_eval_results(self, metrics: Dict[str, Any]):
        # print and write to file
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and (not np.isnan(v)) else str(v)

        # print("Final evaluation results:")
        # for k, v in metrics.items():
        #     if k in ("TP", "TN", "FP", "FN"):
        #         print(f"{k}: {v}")
        #     else:
        #         print(f"{k}: {fmt(v)}")

        eval_file = self.output_path / f"{self.args.model}_{self.args.grad}_{'flatten' if self.args.flatten else 'hier'}_evaluation_results.txt"
        with open(eval_file, 'w') as f:
            for k, v in metrics.items():
                if k != "roc_data":  # Skip roc_data for text file
                    f.write(f"{k}: {v}\n")
        print(f"Saved evaluation -> {eval_file}")
        
        # Save ROC curve data to separate numpy file
        if "roc_data" in metrics:
            roc_file = self.output_path / f"roc_data_{self.args.model}_{self.args.grad}_{'flatten' if self.args.flatten else 'hier'}.npz"
            np.savez(roc_file, 
                     fpr=np.array(metrics["roc_data"]["fpr"]), 
                     tpr=np.array(metrics["roc_data"]["tpr"]), 
                     auc=metrics["roc_data"]["auc"])
            print(f"Saved ROC data -> {roc_file}")

    def _save_model_trace(self):
        # safe trace: many custom modules might not be traceable; do in try/except
        trace_name = f"{self.args.model}_" + (self.args.grad if self.args.model == "sdf" else "") + ("_flatten_" if self.args.flatten else "_hier_") + "net.pt"
        trace_path = self.output_path / trace_name
        try:
            # try to find a reasonable num_joints attribute
            example_dim = getattr(self.net, "num_joints", None)
            if example_dim is None and hasattr(self.net, "encoder"):
                example_dim = getattr(self.net.encoder, "num_joints", None) or getattr(self.net.encoder, "remap_index", None)
                if isinstance(example_dim, (list, tuple, np.ndarray)):
                    example_dim = int(len(example_dim))
            if example_dim is None:
                example_input = torch.zeros(1, self.mj_model.njnt)
            else:
                example_input = torch.zeros(1, int(example_dim))
            example_input = example_input.to('cpu')
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
    parser.add_argument("--data_path", type=str, default="data/universal_robots_ur5e_collision")
    parser.add_argument("--output_path", type=str, default="output/universal_robots_ur5e_collision")
    parser.add_argument("--scene_xml", type=str, default="robot_xml/scene.xml")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--root_name", type=str, default="base")
    parser.add_argument("--batchsize", type=int, default=100000)
    parser.add_argument("--eval_time", type=int)
    parser.add_argument("--without_eikonal", action="store_true")
    parser.add_argument("--model", choices=['sdf', 'ndf', 'lipschitz'], default='sdf')
    parser.add_argument("--grad", choices=["base", "none", "steik"], default='base')
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--no-scalar-s", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for NDF collision detection")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
