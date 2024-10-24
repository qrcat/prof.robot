import json
import torch
import numpy as np
import pathlib
import collections



data_path = pathlib.Path("data/unitree_h1")

iteration = []
collision = []

w_collision = {}
n_collision = {}
for json_path in data_path.glob('sample_*/info.json'):
    with json_path.open() as f:
        data = json.load(f)
    joint_positions = np.load(json_path.parent / "joint_positions.npy")

    info = {
        "joint_positions": joint_positions,
        "collision": data['collision'],
    }


    if data['collision'] > 0:
        w_collision[json_path] = info
    else:
        n_collision[json_path] = info

not_collision_pose = [info['joint_positions'] for path, info in n_collision.items()]
not_collision_pose = torch.tensor(not_collision_pose)

for json_path, info in w_collision.items():
    joint_positions = torch.tensor(info['joint_positions'])
    distances = torch.norm(joint_positions - not_collision_pose, dim=1)
    _, indices = torch.topk(distances, 1, largest=False)
    nearest_neighbors = not_collision_pose[indices]
    with json_path.open("r") as f:
        data = json.load(f)
    with json_path.open("w") as f:
        data["distance"] = distances[indices].item()
        json.dump(data, f)
    break


collision_counter = collections.Counter(collision)
