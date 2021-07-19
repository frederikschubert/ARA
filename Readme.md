# ARA - Automatic Risk Adaption

This is the code accompanying the paper "Towards Automatic Risk Adaptation in Distributional Reinforcement Learning", presented at the Reinforcement Learning for Real Life Workshop at ICML 2021.

## Installation

To install this repo's dependencies, run:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
git submodule update --init --recursive  # to pull dependencies
pip install -e dependencies/pybullet-gym
```

## Training

To start training, execute train_bullet.py with a given config, robot and environment limits, e.g.:

```bash
python train_bullet.py --config configs/bullet/rnd_dsac.yml --walker_id DynamicAntBulletEnv-v0 --slippery --slippery_limits 0.8 1.2 --heavy --heavy_limits 0.8 1.2
```

In configs/bullet, you will find configurations for different agents and ARA mappings.
