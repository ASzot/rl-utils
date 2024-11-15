import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
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
    error_bar_color: Union[str, Tuple[float, float, float, float]] = "black",
    error_lw=2,
    error_capsize=3,
    error_capthick=2,
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
    minor_tick_count: Optional[int] = None,
    xaxis_label_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    bar_value_label_font_size: int = -1,
    figsize: Tuple[float] = (6.4, 4.8),
    xtick_groups: Optional[Dict[str, str]] = None,
    replace_zero: Optional[float] = None,
    value_scaling: float = 1.0,
    color_palette_name="colorblind",
):
    """
    :param group_key: This key defines the "x-axis" of the bar graph. So if you
        want a bar per method, then this should be the method key. If you want a
        "grouped" bar graph where each group has a set of bars, and these groups
        are spaced with with the group names at the bottom, then this should be
        that group key.
    :param name_ordering: Order of the group names on the x-axis. This refers
        to the names of "groups" if you are only plotting a single bar per group,
        then this corresponds to values in `group_key` if there are multiple bars
        clustered within a group, then this refers to `bar_group_key`.
    :param group_name_ordering: Order of the bars within a group. In other
        words, the order of bars that are spaced next to each other for a grouped
        bar graph. This is only relevant when `bar_group_key` is set.
    :param bar_pad: Distance between bar groups
    :param within_group_padding: Distance between bars within bar group.
    :param bar_group_key: How to divide up bars within a group. If you are not
        doing a grouped bar graph then don't set this. For a grouped bar graph this
        defines what the bars "within a group" (meaning next to each other) are. So
        if I want bars to be grouped per benchmark and individual bars per method
        (meaning bars per benchmark are next to each other) I would set
        `group_key=method` and `bar_group_key=benchmark`.
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
    :param minor_tick_count: If set, will create this many minor ticks between
        each major tick, these ticks will automatically be colored in as well.
    :param error_bar_color: Color of the error bar. Either a named color or a
        tuple like (0.2, 0.2, 0.2, 1.0) specifying RGBA.
    :param bar_value_label_font_size: If != -1 then this will render the
        numeric value of the bar above the bar in the plot. This controls the font
        size of that bar value. The values are rounded to the nearest whole
        number and displayed in the center of the bar.
    :param xtick_groups: A mapping from the x-tick label to a label "group".
        The group label is displayed below the x-axis labels.
    :param replace_zero: If specified, this will replace any zero value. This
        is used to give bars that have zero value some height to make them more
        visible.
    :param value_scaling: Multiply all plotted values by this amount.
    """

    if name_ordering is None:
        def_idx = [(k, i) for i, k in enumerate(plot_df[group_key].unique())]
        name_ordering = [x for x, _ in def_idx]
    color_pal = sns.color_palette(color_palette_name)
    if name_colors is None and group_colors is None:
        name_colors = {k: color_pal[i] for i, k in enumerate(name_ordering)}
    if rename_map is None:
        rename_map = {}
    if xaxis_label_colors is None:
        xaxis_label_colors = {}

    plot_df = plot_df.replace("missing", missing_fill_value)
    plot_df = plot_df.replace("error", error_fill_value)
    plot_df[plot_key] = plot_df[plot_key].astype("float")

    if bar_group_key is not None:
        bar_grouped = plot_df.groupby(bar_group_key)
    else:
        bar_grouped = [("all", plot_df)]
    num_grouped = len(bar_grouped)

    bar_width = base_bar_width / num_grouped
    start_x = 0.0
    within_group_spacing = bar_width + within_group_padding

    grouped_bar_data = {k: v for k, v in bar_grouped}

    if group_name_ordering is None:
        group_name_ordering = list(grouped_bar_data.keys())

    fig, ax = plt.subplots(figsize=figsize)
    all_use_x = []
    for bar_group_name in group_name_ordering:
        sub_df = grouped_bar_data[bar_group_name]
        df_avg_y = sub_df.groupby(group_key)[plot_key].mean()
        df_std_y = sub_df.groupby(group_key)[plot_key].std()

        avg_y = []
        std_y = []
        within_group_name_ordering = [n for n in name_ordering if n in df_avg_y.index]
        is_missing = []
        is_error = []
        for name in within_group_name_ordering:
            is_missing.append(df_avg_y.loc[name] == missing_fill_value)
            is_error.append(df_avg_y.loc[name] == error_fill_value)
            avg_y.append(df_avg_y.loc[name] * value_scaling)
            std_y.append(df_std_y.loc[name] * error_scaling)

        if group_colors is None:
            # Bars clustered together will be colored differently
            colors = [name_colors[x] for x in within_group_name_ordering]
            labels = [rename_map.get(x, x) for x in within_group_name_ordering]
        else:
            # All bars clustered together will be colored the same.
            colors = [group_colors[bar_group_name] for _ in within_group_name_ordering]
            labels = rename_map.get(bar_group_name, bar_group_name)

        # Convert to colors if we refer to colors as indices in sns color
        # palette.
        colors = [color_pal[x] if isinstance(x, int) else x for x in colors]

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
                    "ecolor": error_bar_color,
                    "lw": error_lw,
                    "capsize": error_capsize,
                    "capthick": error_capthick,
                },
            }

        if replace_zero is not None:
            should_replace_zero = [y == 0 for y in avg_y]
            avg_y = [replace_zero if y == 0 else y for y in avg_y]
        bars = ax.bar(
            use_x,
            avg_y,
            width=bar_width,
            color=colors,
            align="center",
            alpha=bar_alpha,
            edgecolor=(0, 0, 0, 1.0),
            linewidth=bar_edge_thickness,
            label=labels,
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

            # Render text above the bar.
            if bar_value_label_font_size > 0:
                yval = bar.get_height()
                disp_val = yval
                if replace_zero is not None and should_replace_zero[i]:
                    disp_val -= replace_zero
                ax.text(
                    bar.get_x() + (bar.get_width() / 4),
                    yval,
                    int(round(disp_val, 0)),
                    va="bottom",
                    fontsize=bar_value_label_font_size,
                )

    if show_ticks:
        xtic_names = [rename_map.get(x, x) for x in name_ordering]
    else:
        xtic_names = ["" for x in name_ordering]

    xtic_locs = all_use_x[len(all_use_x) // 2]
    if include_grid:
        ax.grid(which="both", color="lightgray", linestyle="-", axis="y", zorder=-100)
    ax.set(axisbelow=True)
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_names, rotation=xlabel_rot, fontsize=tic_font_size)
    ax.set_ylabel(rename_map.get(plot_key, plot_key), fontsize=axis_font_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_font_size)

    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if bar_value_label_font_size > 0:
        # Adjust the height a bit to account for the labels on top of the bars.
        cur_ylim = ax.get_ylim()
        ax.set_ylim((cur_ylim[0], cur_ylim[1] + 5))

    ax.grid(which="major", color="lightgray", linestyle="--")

    if title != "":
        ax.set_title(title)
    if legend:
        ax.legend(
            fontsize=legend_font_size,
            ncol=legend_n_cols,
        )

    if minor_tick_count is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_count))
    for lab in ax.get_yticklabels():
        lab.set_fontsize(tic_font_size)

    if xtick_groups is not None:
        # Render the x-tick groups.
        groups = [xtick_groups[x] for x in name_ordering]
        group_counts = OrderedDict((g, groups.count(g)) for g in dict.fromkeys(groups))
        cur_bar_idx = 0
        for group_name, count in group_counts.items():
            label_color = xaxis_label_colors.get(name_ordering[cur_bar_idx], None)

            avg_x_pos = np.mean([bars[cur_bar_idx + i].get_x() for i in range(count)])
            cur_bar_idx += count

            ax.text(
                avg_x_pos,
                -15,
                rename_map.get(group_name, group_name),
                ha="center",
                va="top",
                fontsize=tic_font_size,
                color=label_color,
            )

    x_axis = ax.xaxis
    for orig_name, label in zip(name_ordering, x_axis.get_ticklabels()):
        # Color x-tick names
        if orig_name in xaxis_label_colors:
            label.set_color(xaxis_label_colors[orig_name])
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
