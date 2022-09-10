# Reinforcement Learning Research Utilities

A library of helper functions, environment helpers, and experiment management functions for RL research.
* `launcher`: scripts to launch slurm jobs.
* `envs` utility to setup vectorized environments in RL.
    * `envs/pointmass` A toy navigation task implemented in PyTorch.
* `logging` Weights&Biases, CLI, or Tensorboard logging interfaces.
* `models` useful model components for RL policy networks.
* `common`: helpers to manipulate observation and action spaces, standardize policy evaluation, and help with visualizing policy rollouts.
* `plotting`
    * `wb_query`: CLI for extracting information from Weights&Biases.


# Installation
Requires Python >= 3.7.

Install from source for development:
* Clone this repository `git clone https://github.com/ASzot/rl-utils.git`
* `pip install -e .`

# Environments
Toy environments to test algorithms.
* [Point Mass Navigation](https://github.com/ASzot/rl-helper/tree/main/rl_utils/envs/pointmass)
