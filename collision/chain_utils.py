import torch
import mujoco
from pytorch_kinematics.mjcf import _build_chain_recurse
from pytorch_kinematics import chain, frame
import pytorch_kinematics.transforms as tf


from typing import Union


def build_chain_relation_map(path, body: Union[None, str, int] = None):
    model = mujoco.MjModel.from_xml_path(path)
    if body is None:
        root_body = model.body(0)
    else:
        root_body = model.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                            offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(model, root_frame, root_body)


    # build relation
    c = chain.Chain(root_frame)
    remap_input = []
    for frame_idx in c.get_all_frame_indices():
        chain_idxs = c.parents_indices[frame_idx]
        remap_input.append([c.joint_indices[idx].item() for idx in chain_idxs])
    output_list = []
    for item in remap_input:
        while item and item[0] == -1:
            item.pop(0)
        if item:
            if item[-1] == -1: continue
            output_list.append(item)
    return output_list, c
