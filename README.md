# RL Job Launcher and Plotting Utilities

# Installation
Requires Python >= 3.7.

* `pip install rl-exp-utils`
* To install from source: `git clone https://github.com/ASzot/rl-utils.git && cd rl-utils && git install -e .`

# Experiment Launcher
Utility script to launch jobs on Slurm (either via sbatch or srun), in a new tmux window, with PyTorch distributed, or in the current shell.

Examples:
* Launch job in current window using `~/configs/proj.yaml` config: `python -m rl_utils.launcher --cfg ~/configs/proj.yaml python imitation_learning/run.py`
* Launch job on `user-overcap` partition: `python -m rl_utils.launcher --partition user-overcap --cfg ~/configs/proj.yaml python imitation_learning/run.py`
* Evaluate the last checkpoint from a group of runs: `python -m rl_utils.launcher --cfg ~/configs/proj.yaml --proj-dat eval --slurm small python imitation_learning/eval.py load_checkpoint="&last_model WHERE group=Ef6a88c4f&"`

Arguments
* `--pt-proc`: Run with `torchrun` using `--pt-proc` per node.
* `--cd`: Sets the `CUDA_VISIBLE_DEVICES`.
* `--sess-name`: tmux session name to attach to (by default none).
* `--sess-id`: tmux session ID to attach to (by default none).
* `--group-id`: Add a group prefix to the runs.
* `--run-single`: Run several commands sequentially.
* `--time-freq X`: Run with `pyspy` timing at frequency `X`. Will save the profile to `data/profile/scope.speedscope`.

Slurm arguments:
* `--g`: Number of SLURM GPUs.
* `--c`: Number of SLURM CPUs.
* `--comment`: Comment to leave on SLURM run.

## Run Exp Config Schema
Keys in config file.
* `add_all: str`: Suffix that is added to every command.
* `ckpt_cfg_key`: The key to get the checkpoint folder from the config.
* `ckpt_append_name`: If True, the run name is appended to the checkpoint folder.
* `slurm_ignore_nodes: List[str]`: List of Slurm hosts that should be ignored.
* `proj_dat_add_env_vars: Dict[str, str]`: Mapping `--proj-dat` key to environment variables to export. Multiple environment variables are separated by spaces.
* `eval_sys`: Configuration for the evaluation system. More information on this below.

Variables that are automatically substituted into the commands:
* `$GROUP_ID`: A unique generated ID assigned to all runs from the command.
* `$SLURM_ID`: The slurm job name. Randomly generated for every run. This is generated whether the job is running on slurm or not.
* `$DATA_DIR`: `base_data_dir` in the config
* `$CMD_RANK`: The index of the command in the list of commands to run.
* `$PROJECT_NAME`: `proj_name` from config.
* `$WB_ENTITY`: `wb_entity` from config.

Example:
```yaml
base_data_dir: ""
proj_name: ""
wb_entity: ""
ckpt_cfg_key: "CHECKPOINT_FOLDER"
ckpt_append_name: False
add_env_vars:
  - "MY_ENV_VAR=env_var_value"
conda_env: "conda_env_name"
slurm_ignore_nodes: ["node_name"]
add_all: "ARG arg_value"
eval_sys:
  ckpt_load_k: "the argument name to pass the evaluation checkpoint directory to"
  ckpt_search_dir: "folder name relative to base data dir where checkpoints are saved."
  change_vals:
    "arg name": "new arg value"
proj_data:
  option: "ARG arg_value"
slurm:
  profile_name:
    c: 7
    partition: 'partition_name'
    constraint: 'a40'
```

# Auto Evaluation System
Automatically evaluate experiments from the train job slurm launch script. Example usage: `python -m rl_utils.launcher.eval_sys --runs th_im_single_Ja921cfd5 --proj-dat render`. The `eval_sys` config key in the project config specifies how to change the launch command for evaluation (like loading a checkpoint or changing to an evaluation mode).

# Plotting

## Auto Line
Run something like `python -m rl_utils.plotting.auto_line --cfg plot_cfgs/my_plot.yaml` where `plot_cfgs/my_plot.yaml` looks something like:
```yaml
methods:
  "dense_reward": "Ud05e1467"
  "sparse": "W5c609da1"
  "mirl": "bf69a9e1"
method_spec: "group"
proj_cfg: "/Users/andrewszot/configs/mbirlo.yaml"
plot_key: "dist_to_goal"
save_name: "reward"
use_cached: True
plot_params:
  smooth_factor: 0.7
  legend: True
  rename_map:
    "dense_reward": "Dense Reward"
    "sparse": "Sparse Reward"
    "_step": "Step"
    "dist_to_goal": "Distance To Goal"
    "mirl": "Meta-IRL"
```
Argument description:
* `method_spec`: Key to group methods by. For example: `group` name.
* `save_name`: No extension and no parent directory.
* `methods`: Dict of names mapping to `method_spec` instances (for example group names).

## Auto Bar
Config Schema:
```yaml
methods:
  st_pop: K94569d43
  im_pop: Rb2cd0028
method_spec: "group"
proj_cfg: "../../configs/hr.yaml"
plot_key: "eval_reward/average_reward"
save_name: "set_table"
use_cached: False
plot_params: {}
```

# W&B Query
Selectable fields:

- `summary`: The metrics for the model at the end of training. Also the run state. Useful if you want to check run result.
- Any other key: If the key is none of the above, then it will get the relevant key from the `summary` dict in the run (the final value).

# Environments
Toy environments to test algorithms.
* [Point Mass Navigation](https://github.com/ASzot/rl-helper/tree/main/rl_utils/envs/pointmass)
