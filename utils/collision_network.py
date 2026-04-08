import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco as mj

import math


class HyperEnbedding(nn.Module):
    def __init__(
        self,
        n_joint: int,
        feature_dim: int,
        parent_name: list,
        remap_index: list,
        root_frame,
        network_util=nn.Linear,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.remap_index = nn.Parameter(torch.tensor(remap_index), requires_grad=False)
        self.flatten_index = nn.Parameter(
            torch.tensor([i for i in range(n_joint) if i not in remap_index]),
            requires_grad=False,
        )

        # get parents mapping
        parents_map = {}

        def traverse_tree(frame, parent_name, depth=0):
            parents_map[frame.name] = parent_name

            for child in frame.children:
                traverse_tree(child, frame.name, depth + 1)

        traverse_tree(root_frame, None)

        moduled_dict = {}

        if len(remap_index) < n_joint:
            flatten_dim = n_joint - len(remap_index)
            moduled_dict["flatten"] = network_util(
                flatten_dim, feature_dim * flatten_dim
            )

        forward_code = "def forward(self, x):"

        for index, name in zip(remap_index, parent_name):
            node = name

            while node is not None:
                node = parents_map[node]
                if node in parent_name:  # this variable already exists
                    break

            if node is None:
                moduled_dict[name] = network_util(1, self.feature_dim)
                forward_code += (
                    f"\n    {name} = self.module['{name}'](x[:, {index}:{index+1}])"
                )
            else:
                moduled_dict[name] = network_util(
                    self.feature_dim + 1, self.feature_dim
                )
                forward_code += f"\n    {name} = self.module['{name}'](torch.cat([{node}, x[:, {index}:{index+1}]], dim=-1))"

        if moduled_dict.get("flatten") is not None:
            forward_code += (
                f"\n    flatten = self.module['flatten'](x[:, self.flatten_index])"
            )
            forward_code += (
                "\n    return torch.cat(["
                + ", ".join(parent_name)
                + ", flatten], dim=-1)"
            )
        else:
            forward_code += (
                "\n    return torch.cat([" + ", ".join(parent_name) + "], dim=-1)"
            )

        self.module = nn.ModuleDict(moduled_dict)

        self.forward_code = forward_code

        exec(self.forward_code + "\nself.forward = forward.__get__(self)")

    @staticmethod
    def build_remap_index(mj_model, pk_chain):
        num_joints = mj_model.njnt

        model_joints = {}

        for i in range(num_joints):
            joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, i)

            qpos_index = mj_model.jnt_qposadr[i]

            model_joints[joint_name] = qpos_index

        return [model_joints[joint] for joint in pk_chain.get_joint_parameter_names()]


class AdditiveAttentionPool(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (B, N, C)
        return: (B, C)
        """
        # attention scores
        score = self.attn(x)              # (B, N, 1)
        weight = torch.softmax(score, 1)  # (B, N, 1)

        # weighted sum
        return (weight * x).sum(dim=1)


class PointNet(nn.Module):
    """PointNet for point cloud feature extraction.

    Args:
        output_dim: Output feature dimension C
        hidden_dim: Hidden dimension for MLPs
        use_bn: Whether to use batch normalization
    """

    def __init__(self, output_dim=256, hidden_dim=64, use_bn=True):
        super().__init__()

        self.output_dim = output_dim
        self.use_bn = use_bn

        # Point feature extraction MLP (shared across all points)
        if use_bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.LeakyReLU(),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
            )

        # Global feature extraction MLP
        if use_bn:
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim * 2, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(),
            )
        else:
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim * 2, output_dim),
                nn.LeakyReLU(),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Point cloud tensor of shape (B, N, 3)

        Returns:
            Global feature tensor of shape (B, C)
        """
        batch_size, num_points, _ = x.shape

        # Reshape for MLP: (B, N, 3) -> (B*N, 3)
        x = x.view(-1, 3)

        # Extract per-point features: (B*N, 3) -> (B*N, hidden_dim*2)
        x = self.mlp1(x)

        # Pooling
        x = x.view(batch_size, num_points, -1)
        x = x.max(dim=1)[0]

        # Extract global features: (B, hidden_dim*2) -> (B, output_dim)
        x = self.mlp2(x)

        return x


class HyperNetwork(nn.Module):
    def __init__(
        self,
        num_joints,
        parent_name,
        remap_index,
        root_frame,
        feature_dim=8,
        hidden_dim=512,
        output_dim=1,
        init_s=2.3,
        hidden_activation="LeakyReLU",
        flatten=False,
    ):
        super().__init__()
        self.num_joints = num_joints

        if flatten:
            self.encoder = nn.Linear(num_joints, num_joints * feature_dim)
        else:
            self.encoder = HyperEnbedding(
                num_joints, feature_dim, parent_name, remap_index, root_frame
            )

        self._mlp = nn.Sequential(
            getattr(nn, hidden_activation)(),
            nn.Linear(self.num_joints * feature_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, output_dim),
        )

        s = torch.tensor([init_s])

        if init_s == 0.0:
            self.register_buffer('s', s)
        else:
            self.s = nn.Parameter(s, requires_grad=True)

    def forward(self, x):
        x = self.encoder(x)

        return self._mlp(x), self.s.exp().clamp(0.001, 1000)


class HyperDynamicNetwork(nn.Module):
    def __init__(
        self,
        num_joints,
        parent_name,
        remap_index,
        root_frame,
        feature_dim=8,
        hidden_dim=512,
        output_dim=1,
        init_s=2.3,
        hidden_activation="LeakyReLU",
        flatten=False,
    ):
        super().__init__()
        self.num_joints = num_joints

        if flatten:
            self.encoder = nn.Linear(num_joints, num_joints * feature_dim)
        else:
            self.encoder = HyperEnbedding(
                num_joints, feature_dim, parent_name, remap_index, root_frame
            )

        self.pointnet = PointNet(output_dim=256, hidden_dim=64, use_bn=True)

        self._mlp = nn.Sequential(
            getattr(nn, hidden_activation)(),
            nn.Linear(self.num_joints * feature_dim + 256, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.s = nn.Parameter(torch.tensor([init_s], requires_grad=True))

    def forward(self, x, pc):
        x = self.encoder(x)
        y = self.pointnet(pc)

        return self._mlp(torch.cat([x, y], dim=1)), self.s.exp().clamp(0.001, 1000)


class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), requires_grad=True)
        )
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()  # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(
            input, self.weight * scale.unsqueeze(1), self.bias
        )

    def __repr__(self):
        return "LipschitzLinear(in_features={}, out_features={})".format(
            self.in_features, self.out_features
        )


# From https://github.com/whitneychiu/lipmlp_pytorch/blob/main/2D_interpolation/main_lipmlp.py
class LipMLP(torch.nn.Module):
    def __init__(self, dims, init_s=2.3):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.num_joints = dims[0]

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims) - 2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii + 1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

        self.s = nn.Parameter(torch.tensor([init_s], requires_grad=True))

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc * self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x), torch.exp(self.s).clamp(0.001, 1000)
