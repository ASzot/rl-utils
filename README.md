# RL-Helper
A library of useful components for RL research.
* `envs` utility to setup vectorized environments.
    * `envs/pointmass` A toy navigation task implemented in PyTorch.
* `logging` run CLI, Wandb, or Tensorboard logging.
* `models` useful model components for RL policy networks.
* `common`  helpers to manipulate observation and action spaces, standardize policy evaluation, and help with visualizing policy rollouts.

# Installation
Install from source for development:
* Clone this repository `git clone https://github.com/ASzot/rl-helper.git`
* `pip install -e .`

# Environments
* [Point Mass Navigation](https://github.com/ASzot/rl-helper/tree/main/rl_helper/envs/pointmass): A 2D point navigates to the goal.
