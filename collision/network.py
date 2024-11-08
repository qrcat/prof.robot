import torch
import torch.nn as nn


class SingleNetwork(nn.Module):
    def __init__(self, num_joints, hidden_dim=512, 
                 hidden_activation='LeakyReLU'):
        super().__init__()

        self._mlp = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, 1)
        )

        self.s = nn.Parameter(torch.tensor([10.0], requires_grad=True))

    def forward(self, x):
        return self._mlp(x), self.s.clamp(0.001, 1000)


class HyperEnbedding(nn.Module):
    def __init__(self, feature_dim: int, tree: list):
        super().__init__()
        self.tree = tree

        modules = []
        for i in self.tree:
            input_dim = feature_dim + 1 if len(i) > 1 else 1
            modules.append(nn.Linear(input_dim, feature_dim))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        embedded_list = []
        embedded_dict = {}

        for i, map_rel in enumerate(self.tree):
            input_embed = x[:, map_rel[-1:]]

            if map_rel[:-1]:
                embed_input = torch.cat([embedded_dict[tuple(map_rel[:-1])], input_embed], dim=1)
            else:
                embed_input = input_embed

            embedded_i = self.modules_list[i](embed_input)

            embedded_list.append(embedded_i)
            embedded_dict[tuple(map_rel)] = embedded_i
    
        return torch.cat(embedded_list, dim=1)


class HyperNetwork(nn.Module):
    def __init__(self, num_joints, mapping_relations, feature_dim=8, hidden_dim=512,
                 hidden_activation='LeakyReLU'):
        super().__init__()
        self.num_joints = num_joints

        self.hyper = HyperEnbedding(feature_dim, mapping_relations)

        self._mlp = nn.Sequential(
            getattr(nn, hidden_activation)(),
            nn.Linear(self.num_joints*feature_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, hidden_dim),
            getattr(nn, hidden_activation)(),
            nn.Linear(hidden_dim, 1),
        )

        self.s = nn.Parameter(torch.tensor([10.0], requires_grad=True))

    def forward(self, x):
        x = self.hyper(x)

        return self._mlp(x), self.s.clamp(0.001, 1000)
