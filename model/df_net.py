from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Tuple
import json
import pathlib


class DrRobotData(data.Dataset):
    def __init__(self, data_path="data/unitree_h1") -> None:
        self.data_path = pathlib.Path(data_path)
        self.data = []
        for json_path in self.data_path.glob('sample_*/info.json'):
            with json_path.open() as f:
                data = json.load(f)
            distance = data.get('distance', 0)
            joint_positions = np.load(json_path.parent / "joint_positions.npy")
            self.data.append(
                (torch.tensor(joint_positions, dtype=torch.float), torch.tensor([distance], dtype=torch.float))
                )


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class DFNet(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self._df_net = nn.Sequential(
            nn.Linear(num_joints, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.requires_grad_(True)
        dist = self._df_net(x)
        try:
            d_points = torch.ones_like(dist).requires_grad_(False)
            # TODO: add eikonal loss here
            grad_val = torch.autograd.grad(
                outputs=dist,
                inputs=x,
                grad_outputs=d_points,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            eikonal_loss = ((grad_val.norm(2, dim=-1) - 1) ** 2)
        except:
            eikonal_loss = torch.zeros_like(x[:, :1])
        return dist, {"eikonal": eikonal_loss}
        

dataset = DrRobotData()
ds_train, ds_test = torch.utils.data.random_split(dataset, [0.75, 0.25])
dl_train = data.DataLoader(ds_train, batch_size=100, shuffle=True)
dl_test = data.DataLoader(ds_test, batch_size=100, shuffle=True)


model = DFNet(19)
model.cuda()

optim = torch.optim.Adam(model.parameters(), lr=1e-5)

writer = SummaryWriter('output/unitree_h1/dfnet')

tbar = trange(10000)
for epoch in tbar:
    ...
    for joint_positions, distance in dl_train:
        dist, loss_item = model(joint_positions.cuda())

        Ldist = torch.nn.functional.l1_loss(dist, distance.cuda())
        
        Leikonal = loss_item["eikonal"].mean()

        loss = Ldist + 0.5 * Leikonal

        loss.backward()

        optim.step()
        optim.zero_grad()

    writer.add_scalar('train/dist', Ldist, epoch)
    writer.add_scalar('train/eikonal', Leikonal, epoch)
    
    with torch.no_grad():
        Ldist = []
        for joint_positions, distance in dl_test:
            dist, _ = model(joint_positions.cuda())
            Ldist.append(dist-distance.cuda())
        Ldist = torch.cat(Ldist).abs().mean()
        writer.add_scalar('test/dist', Ldist, epoch)

    tbar.set_postfix({
        "loss": format(loss.item(), ".4f"),
        "dist": format(Ldist.item(), ".4f"),
        "eikonal": format(Leikonal.item(), ".4f"),
    })

torch.save(model.state_dict(), 'output/unitree_h1/df_net.ckpt')

print(1)

