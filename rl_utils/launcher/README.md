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

Variables that are automatically substituted into the commands:
* `$GROUP_ID`: A unique generated ID assigned to all runs from the command.
* `$SLURM_ID`: The slurm job name. Randomly generated for every run. This is generated whether the job is running on slurm or not.
* `$DATA_DIR`: `base_data_dir` in the config
* `$CMD_RANK`: The index of the command in the list of commands to run.
* `$PROJECT_NAME`: `proj_name` from config.
* `$WB_ENTITY`: `wb_entity` from config.

Example:
```
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
proj_data:
  option: "ARG arg_value"
slurm:
  profile_name:
    c: 7
    partition: 'partition_name'
    constraint: 'a40'
```



## WB
Dynamically substitute values into your commands by surrounding
