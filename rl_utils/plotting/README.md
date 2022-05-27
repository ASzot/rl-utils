## Auto Plotter
Run something like `python -m rl_utils.plotting.auto_line --cfg plot_cfgs/my_plot.yaml` where `plot_cfgs/my_plot.yaml` looks something like:
```
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
