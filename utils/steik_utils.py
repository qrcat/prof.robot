import numpy as np
import torch
import torch.nn as nn

# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
import numpy as np
import torch
from torch.autograd import grad


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type="abs"):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == "abs":
        eikonal_term = ((all_grads.norm(2, dim=-1) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=-1) - 1).square()).mean()

    return eikonal_term

def center_and_scale(points, cp=None, scale=None):
    # center a point cloud and scale it to unite sphere.
    if cp is None:
        cp = points.mean(axis=1)
    points = points - cp[:, None, :]
    if scale is None:
        scale = np.linalg.norm(points, axis=-1).max(-1)
    points = points / scale[:, None, None]
    return points, cp, scale


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[
        0
    ]  # [:, -3:]
    return points_grad


def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = (
        0.5
        * torch.autograd.grad(
            dot_grad, points, hvp, retain_graph=True, create_graph=True
        )[0]
    )
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads**2, dim=-1) + 1e-5)
    return div


def full_div(points, grads):
    dx = gradient(points, grads[:, :, 0])
    dy = gradient(points, grads[:, :, 1])
    if points.shape[-1] == 3:
        dz = gradient(points, grads[:, :, 2])
        div = dx[:, :, 0] + dy[:, :, 1] + dz[:, :, 2]
    else:
        div = dx[:, :, 0] + dy[:, :, 1]
    div[div.isnan()] = 0
    return div


class StEikLoss(nn.Module):
    def __init__(
        self,
        weights=[3e3, 1e2, 1e2, 5e1, 1e2],
        loss_type="siren",
        div_decay="none",
        div_type="dir_l1",
    ):
        super().__init__()
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_div = True if "div" in self.loss_type else False

    def forward(self, nonmnfld_points, nonmanifold_pnts_pred, remap_index):
        #########################################
        # Compute required terms
        #########################################
        div_loss = torch.tensor([0.0], device=nonmnfld_points.device)

        nonmnfld_grad = gradient(nonmnfld_points, nonmanifold_pnts_pred)[:, remap_index]

        # div_term
        if self.use_div and self.weights[4] > 0.0:

            if self.div_type == "full_l2":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(
                    torch.square(nonmnfld_divergence), 0.1, 50
                )
            elif self.div_type == "full_l1":
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(
                    torch.abs(nonmnfld_divergence), 0.1, 50
                )
            elif self.div_type == "dir_l2":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence)
            elif self.div_type == "dir_l1":
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence)
            else:
                raise Warning(
                    "unsupported divergence type. only suuports dir_l1, dir_l2, full_l1, full_l2"
                )

            div_loss = nonmnfld_divergence_term.mean()  # + mnfld_divergence_term.mean()

        # eikonal term
        eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=None, eikonal_type="abs")

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(nonmanifold_pnts_pred)).mean()

        #########################################
        # Losses
        #########################################

        # losses used in the paper
        loss = self.weights[1] * inter_term + self.weights[3] * eikonal_term

        return {
            "loss": loss,
            "inter_term": inter_term,
            "eikonal_term": eikonal_term,
            "div_loss": div_loss,
        }

    def update_div_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, "decay_params_list"):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(
                zip(
                    [params[0], *params[1:-1][1::2], params[-1]],
                    [0, *params[1:-1][::2], 1],
                )
            )

        curr = current_iteration / n_iterations
        we, e = min(
            [tup for tup in self.decay_params_list if tup[1] >= curr],
            key=lambda tup: tup[1],
        )
        w0, s = max(
            [tup for tup in self.decay_params_list if tup[1] <= curr],
            key=lambda tup: tup[1],
        )

        # Divergence term anealing functions
        if self.div_decay == "linear":  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            elif (
                current_iteration >= s * n_iterations
                and current_iteration < e * n_iterations
            ):
                self.weights[4] = w0 + (we - w0) * (
                    current_iteration / n_iterations - s
                ) / (e - s)
            else:
                self.weights[4] = we
        elif (
            self.div_decay == "quintic"
        ):  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            elif (
                current_iteration >= s * n_iterations
                and current_iteration < e * n_iterations
            ):
                self.weights[4] = w0 + (we - w0) * (
                    1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5
                )
            else:
                self.weights[4] = we
        elif self.div_decay == "step":  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[4] = w0
            else:
                self.weights[4] = we
        elif self.div_decay == "none":
            pass
        else:
            raise Warning("unsupported div decay value")
