<div align="center">

# Prof. Robot: Differentiable Robot Rendering Without Static and Self-Collisions

### CVPR 2025 (Poster)
[Quanyuan Ruan†](https://qrcat.github.io/)<sup>1</sup>, [Jiabao Lei†](https://jblei.site/)<sup>1,2</sup>, Wenhao Yuan<sup>1</sup>, [Yanglin Zhang](https://github.com/lucky9-cyou/)<sup>1,2</sup>, Dekun Lu<sup>1</sup>, [Guiliang Liu*](http://guiliang.me/)<sup>1,2</sup>, [Kui Jia*](http://kuijia.site/)<sup>1,2</sup>

<sup>1</sup>South China University of Technology, <sup>2</sup>School of Data Science, The Chinese University of Hong Kong, Shenzhen

<sup>†</sup> Equal Contribution, <sup>*</sup> Corresponding authors

| Update
| 2026.02.15: We update Dynamic scene of Prof. Robot for research. 

<div>
<button style="background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-right: 10px;" onclick="window.location.href='https://arxiv.org/abs/2503.11269'"><strong>Paper</strong></button>
<button style="background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-right: 10px;" onclick="window.location.href='https://github.com/qrcat/prof.robot'"><strong>Code</strong></button>
<button style="background-color: #ffc107; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-right: 10px;" onclick="window.location.href='/prof-robot'"><strong>Project</strong></button>
</div>

<div align="center">
  <img src="assets/teaser-prof.robot.png" style="width:80%" />
</div>

</div>

<br>


This is the official repository for Prof. Robot, forking from Differentiable Robot Rendering.

## Setup

```bash
conda create -n dr python=3.10 -y
conda activate dr
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install gcc=9 gxx=9 -c conda-forge
pip install gsplat tensorboard ray tqdm mujoco open3d plyfile pytorch-kinematics random-fourier-features-pytorch pytz gradio
```

if `gsplat` installation fails, please reference [gsplat](https://github.com/nerfstudio-project/gsplat).

## Usage

generate collision dataset:

```bash
python generate_robot_collision_data.py --model_xml_dir mujoco_menagerie/universal_robots_ur5e --root_name base
```

train

```bash
python train_collision.py --data_path data/universal_robots_ur5e_collision --output_path output/universal_robots_ur5e_collision --root_name base
```

## Dynamic

```bash
python -m script.generate_robot_dynamic_data
```

```bash
python -m script.train_dynamic_collision
```

## Acknowledgements 🙏

- [Dr. Robot](https://github.com/cvlab-columbia/drrobot)
