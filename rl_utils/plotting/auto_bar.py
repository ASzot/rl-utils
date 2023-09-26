import argparse
from typing import Dict, List, Optional, Tuple

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
    plot_df: pd.DataFrame,
    group_key: str,
    plot_key: str,
    name_ordering: Optional[str] = None,
    group_name_ordering: Optional[List[str]] = None,
    name_colors=None,
    rename_map: Optional[Dict[str, str]] = None,
    show_ticks: bool = True,
    tic_font_size: int = 14,
    axis_font_size: int = 14,
    legend_font_size: int = 14,
    y_disp_bounds: Tuple[float, float] = None,
    title: str = "",
    error_scaling=1.0,
    missing_fill_value: float = MISSING_VAL,
    error_fill_value: float = ERROR_VAL,
    bar_group_key: Optional[str] = None,
    base_bar_width: float = 0.35,
    bar_darkness: float = 0.2,
    bar_alpha: float = 0.9,
    bar_pad: float = 0.2,
    within_group_padding: float = 0.01,
    group_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    legend: bool = False,
    xlabel: Optional[str] = None,
    xlabel_rot: int = 30,
    include_err: bool = True,
    include_grid: bool = False,
    bar_edge_thickness: float = 0.0,
    legend_n_cols: int = 1,
    xaxis_label_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
):
    """
    :param group_key: The key to take the average/std over. Likely the method key.
    :param name_ordering: Order of the group names on the x-axis.
    :param group_name_ordering: Order of the bars within a group.
    :param bar_pad: Distance between bar groups
    :param within_group_padding: Distance between bars within bar group.
    :param bar_group_key: Group columns next to each other.
    :param base_bar_width: Bar width. Scaled by the # of bars per group.
    :param group_colors: Maps the bar group key to a color (RGB float tuple
        [0,1]). Overrides `name_colors`.
    :param xlabel_rot: The rotation (in degrees) of the labels on the x-axis.
    :param include_err: Whether to include error bars.
    :param include_grid: Whether to render a background grid behind the bars.
    :param bar_edge_thickness: How thick the border edges of each bar are.
    :param legend_n_cols: Number of columns in the legend (if it is displayed).
    :param xaxis_label_colors: Specifies the colors of the x-axis labels. Maps
        from the original x-axis group name, not the renamed version.
    """

    def_idx = [(k, i) for i, k in enumerate(plot_df[group_key].unique())]
    if name_ordering is None:
        name_ordering = [x for x, _ in def_idx]
    colors = sns.color_palette()
    if name_colors is None and group_colors is None:
        name_colors = {k: colors[v] for k, v in def_idx}
    if rename_map is None:
        rename_map = {}
    if xaxis_label_colors is None:
        xaxis_label_colors = {}

    plot_df = plot_df.replace("missing", missing_fill_value)
    plot_df = plot_df.replace("error", error_fill_value)
    plot_df[plot_key] = plot_df[plot_key].astype("float")

    bar_grouped = plot_df.groupby(bar_group_key)
    num_grouped = len(bar_grouped)

    bar_width = base_bar_width / num_grouped
    start_x = 0.0
    within_group_spacing = bar_width + within_group_padding

    grouped_bar_data = {k: v for k, v in bar_grouped}

    if group_name_ordering is None:
        group_name_ordering = list(grouped_bar_data.keys())

    fig, ax = plt.subplots()
    all_use_x = []
    for bar_group_name in group_name_ordering:
        sub_df = grouped_bar_data[bar_group_name]
        df_avg_y = sub_df.groupby(group_key)[plot_key].mean()
        df_std_y = sub_df.groupby(group_key)[plot_key].std()

        avg_y = []
        std_y = []
        group_name_ordering = [n for n in name_ordering if n in df_avg_y.index]
        is_missing = []
        is_error = []
        for name in group_name_ordering:
            is_missing.append(df_avg_y.loc[name] == missing_fill_value)
            is_error.append(df_avg_y.loc[name] == error_fill_value)
            avg_y.append(df_avg_y.loc[name])
            std_y.append(df_std_y.loc[name] * error_scaling)
        if group_colors is None:
            colors = [name_colors[x] for x in group_name_ordering]
        else:
            colors = [group_colors[bar_group_name] for _ in group_name_ordering]

        N = len(avg_y)
        end_x = round(start_x + N * (bar_width + bar_pad), 3)

        use_x = np.linspace(start_x, end_x, N)
        all_use_x.append(use_x)

        kwargs = {}
        if include_err:
            std_y = np.nan_to_num(std_y, nan=0.0)

            kwargs = {
                "yerr": std_y,
                "error_kw": {
                    "ecolor": (bar_darkness, bar_darkness, bar_darkness, 1.0),
                    "lw": 2,
                    "capsize": 3,
                    "capthick": 2,
                },
            }

        use_name = rename_map.get(bar_group_name, bar_group_name)
        bars = ax.bar(
            use_x,
            avg_y,
            width=bar_width,
            color=colors,
            align="center",
            alpha=bar_alpha,
            edgecolor=(0, 0, 0, 1.0),
            linewidth=bar_edge_thickness,
            label=use_name,
            **kwargs,
        )
        start_x += within_group_spacing
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

    xtic_locs = all_use_x[len(all_use_x) // 2]
    if include_grid:
        ax.grid(which="major", color="lightgray", linestyle="-", axis="y", zorder=-100)
    ax.set(axisbelow=True)
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_names, rotation=xlabel_rot, fontsize=tic_font_size)
    ax.set_ylabel(rename_map.get(plot_key, plot_key), fontsize=axis_font_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_font_size)
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if title != "":
        ax.set_title(title)
    if legend:
        ax.legend(
            fontsize=legend_font_size,
            ncol=legend_n_cols,
        )
    for lab in ax.get_yticklabels():
        lab.set_fontsize(tic_font_size)

    x_axis = ax.xaxis
    new_name_to_orig = {v: k for k, v in rename_map.items()}
    for label in x_axis.get_ticklabels():
        new_name = label.get_text()
        orig_name = new_name_to_orig.get(new_name, new_name)
        if orig_name in xaxis_label_colors:
            color = np.array(xaxis_label_colors[orig_name], dtype=np.float32)
            if sum(color) > 3:
                color /= 255.0
            label.set_color(color)
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
