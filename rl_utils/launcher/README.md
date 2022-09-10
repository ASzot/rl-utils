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

## Config Schema
```
base_data_dir: "/srv/share/aszot3/hr"
proj_name: "hr"
wb_entity: "aszot"
add_env_vars:
  - "MAGNUM_LOG=quiet"
  - "HABITAT_SIM_LOG=quiet"
conda_env: "hr"
slurm_ignore_nodes: ["marvin"]
add_all: "WB.ENTITY $WB_ENTITY WB.RUN_NAME $SLURM_ID WB.PROJECT_NAME $PROJECT_NAME CHECKPOINT_FOLDER $DATA_DIR/checkpoints/$SLURM_ID/ VIDEO_DIR $DATA_DIR/vids/$SLURM_ID/ LOG_FILE $DATA_DIR/logs/$SLURM_ID.log TENSORBOARD_DIR $DATA_DIR/tb/$SLURM_ID/"
proj_data:
  hab_eval: "TEST_EPISODE_COUNT 10 SENSORS \"('THIRD_RGB_SENSOR', 'HEAD_DEPTH_SENSOR')\" TASK_CONFIG.SIMULATOR.DEBUG_RENDER True TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS \"('HEAD_DEPTH_SENSOR', 'THIRD_RGB_SENSOR')\""
  hab_eval_debug: "TEST_EPISODE_COUNT 1 SENSORS \"('THIRD_RGB_SENSOR', 'HEAD_DEPTH_SENSOR')\" TASK_CONFIG.SIMULATOR.DEBUG_RENDER True TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS \"('HEAD_DEPTH_SENSOR', 'THIRD_RGB_SENSOR')\""
  hab_eval_rgb: "TEST_EPISODE_COUNT 10 SENSORS \"('THIRD_RGB_SENSOR', 'HEAD_RGB_SENSOR')\" TASK_CONFIG.SIMULATOR.DEBUG_RENDER True TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS \"('HEAD_RGB_SENSOR', 'THIRD_RGB_SENSOR')\""
  no_render: "VIDEO_OPTION \"()\""
  debug: "NUM_ENVIRONMENTS 1 WRITER_TYPE tb RL.PPO.num_mini_batch 1 LOG_INTERVAL 1"
  im: "--exp-config ma_habitat_baselines/config/hab/tp_srl_multi_skills_im.yaml"
  tb: "WRITER_TYPE tb"
  train_debug: "WRITER_TYPE tb LOG_INTERVAL 1"
slurm:
  def:
    c: 7
    st: user-overcap
  large:
    c: 16
    st: user-overcap
    constraint: 'a40'
```
