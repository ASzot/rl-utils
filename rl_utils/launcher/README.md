Utility script to launch jobs on Slurm (either via sbatch or srun), in a new tmux window, with PyTorch distributed, or in the current shell.

## Examples
* Launch job in current window using `~/configs/proj.yaml` config: `python -m rl_utils.launcher --cfg ~/configs/proj.yaml python imitation_learning/run.py`
* Launch job on `user-overcap` partition: `python -m rl_utils.launcher --st user-overcap --cfg ~/configs/proj.yaml python imitation_learning/run.py`

## Run Exp Launcher

Keys in config file.
* `add_all: str`: Suffix that is added to every command.
* `slurm_ignore_nodes: List[str]`: List of Slurm hosts that should be ignored.

Variables that are automatically substituted into the commands:
* `$GROUP_ID`: A unique generated ID assigned to all runs from the command.
* `$SLURM_ID`: The slurm job name. Randomly generated for every run. This is generated whether the job is running on slurm or not.
* `$DATA_DIR`: `base_data_dir` in the config
* `$CMD_RANK`: The index of the command in the list of commands to run.
* `$PROJECT_NAME`: `proj_name` from config.
* `$WB_ENTITY`: `wb_entity` from config.
