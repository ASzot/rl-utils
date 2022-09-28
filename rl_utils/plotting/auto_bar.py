import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from rl_utils.plotting.utils import fig_save
from rl_utils.plotting.wb_query import batch_query

MISSING_VAL = 0.24444
ERROR_VAL = 0.3444


def plot_bar(
    plot_df,
    group_key,
    plot_key,
    name_ordering=None,
    name_colors=None,
    rename_map=None,
    show_ticks=True,
    axis_font_size=14,
    y_disp_bounds: Tuple[float, float] = None,
    title="",
    error_scaling=1.0,
    missing_fill_value=MISSING_VAL,
    error_fill_value=ERROR_VAL,
):

    def_idx = [(k, i) for i, k in enumerate(plot_df[group_key].unique())]
    if name_ordering is None:
        name_ordering = [x for x, _ in def_idx]
    colors = sns.color_palette()
    if name_colors is None:
        name_colors = {k: colors[v] for k, v in def_idx}
    if rename_map is None:
        rename_map = {}

    plot_df = plot_df.replace("missing", missing_fill_value)
    plot_df = plot_df.replace("error", error_fill_value)
    plot_df[plot_key] = plot_df[plot_key].astype("float")

    df_avg_y = plot_df.groupby(group_key).mean()
    df_std_y = plot_df.groupby(group_key).std()

    avg_y = []
    std_y = []
    name_ordering = [n for n in name_ordering if n in df_avg_y.index]
    is_missing = []
    is_error = []
    for name in name_ordering:
        is_missing.append(df_avg_y[plot_key].loc[name] == missing_fill_value)
        is_error.append(df_avg_y[plot_key].loc[name] == error_fill_value)
        avg_y.append(df_avg_y.loc[name][plot_key])
        std_y.append(df_std_y.loc[name][plot_key] * error_scaling)

    bar_width = 0.35
    bar_darkness = 0.2
    bar_alpha = 0.9
    bar_pad = 0.0
    use_x = np.arange(len(name_ordering))
    colors = [name_colors[x] for x in name_ordering]

    N = len(avg_y)
    start_x = 0.0
    end_x = round(start_x + N * (bar_width + bar_pad), 3)

    use_x = np.linspace(start_x, end_x, N)

    fig, ax = plt.subplots()

    bars = ax.bar(
        use_x,
        avg_y,
        width=bar_width,
        color=colors,
        align="center",
        alpha=bar_alpha,
        yerr=std_y,
        edgecolor=(0, 0, 0, 1.0),
        error_kw={
            "ecolor": (bar_darkness, bar_darkness, bar_darkness, 1.0),
            "lw": 2,
            "capsize": 3,
            "capthick": 2,
        },
    )
    for i, bar in enumerate(bars):
        if is_missing[i]:
            missing_opacity = 0.1
            # prev_color = bar.get_facecolor()
            bar.set_edgecolor((1, 0, 0, missing_opacity))
            bar.set_hatch("//")
        elif is_error[i]:
            missing_opacity = 0.1
            # prev_color = bar.get_facecolor()
            bar.set_edgecolor((0, 0, 1, missing_opacity))
            bar.set_hatch("//")

    if show_ticks:
        xtic_names = [rename_map.get(x, x) for x in name_ordering]
    else:
        xtic_names = ["" for x in name_ordering]

    xtic_locs = use_x
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_names, rotation=30)
    ax.set_ylabel(rename_map.get(plot_key, plot_key), fontsize=axis_font_size)
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if title != "":
        ax.set_title(title)
    return fig


def plot_from_file(plot_cfg_path, add_query_fields=None):
    cfg = OmegaConf.load(plot_cfg_path)
    if add_query_fields is None:
        add_query_fields = []

    query_k = cfg.plot_key

    result = batch_query(
        [[query_k, *add_query_fields] for _ in cfg.methods],
        [{cfg.method_spec: v} for v in cfg.methods.values()],
        all_should_skip=[len(v) == 0 for v in cfg.methods.values()],
        all_add_info=[{"method": k} for k in cfg.methods.keys()],
        proj_cfg=OmegaConf.load(cfg.proj_cfg),
        use_cached=cfg.use_cached,
        verbose=False,
    )
    df = pd.DataFrame(result)
    fig = plot_bar(df, "method", query_k, **cfg.plot_params)
    fig_save("data/vis", cfg.save_name, fig)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    plot_from_file(args.cfg)
