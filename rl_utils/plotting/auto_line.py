import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from rl_utils.plotting.utils import combine_dicts_to_df, fig_save
from rl_utils.plotting.wb_query import batch_query

MARKER_ORDER = ["^", "<", "v", "d", "s", "x", "o", ">"]


def smooth_arr(scalars: List[float], weight: float) -> List[float]:
    """
    Taken from the answer here https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar

    :param weight: Between 0 and 1.
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def make_steps_match(plot_df, group_key, x_name):
    all_dfs = []
    for _, method_df in plot_df.groupby([group_key]):
        grouped_runs = method_df.groupby(["run"])
        max_len = -1
        max_step_idxs = None
        for _, run_df in grouped_runs:
            if len(run_df) > max_len:
                max_len = len(run_df)
                max_step_idxs = run_df[x_name]
        for _, run_df in grouped_runs:
            run_df[x_name] = max_step_idxs[: len(run_df)]
            all_dfs.append(run_df)
    return pd.concat(all_dfs)


def line_plot(
    plot_df: pd.DataFrame,
    x_name: str,
    y_name: str,
    avg_key: str,
    group_key: str,
    smooth_factor: Union[Dict[str, float], float] = 0.0,
    ax: Optional[matplotlib.axes.Axes] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    y_disp_bounds: Optional[Tuple[float, float]] = None,
    x_disp_bounds: Optional[Tuple[float, float]] = None,
    group_colors: Optional[Dict[str, int]] = None,
    override_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    xtick_fn=None,
    ytick_fn=None,
    legend: bool = False,
    rename_map: Optional[Dict[str, str]] = None,
    title=None,
    axes_font_size=14,
    title_font_size=18,
    legend_font_size="x-large",
    method_idxs: Optional[Dict[str, int]] = None,
    num_marker_points: Optional[Dict[str, int]] = None,
    line_styles: Optional[Dict[str, str]] = None,
    tight=False,
    nlegend_cols=1,
    fetch_std=False,
    y_logscale=False,
    x_logscale=False,
    legend_loc: Optional[str] = None,
    ax_dims: Tuple[int, int] = (5, 4),
    marker_size: int = 8,
    marker_order: Optional[List[str]] = None,
):
    """
    :param plot_df: The data to plot. The `avg_key`, `group_key`, `x_name`, and `y_name` all refer to columns in this dataframe. An example dataframe:
        ```
             method  steps  value  seed
        0   method0      0   0.00     0
        1   method0      1   0.25     0
        2   method0      2   0.50     0
        3   method0      3   0.75     0
        4   method1      0   0.00     0
        5   method1      1   0.50     0
        ```
        Where we might set `x_name=steps`, `y_name=value`, `avg_key=seed`,
        `group_key=method`. This would produce two lines with no error shading.
        To get the error shading you would need multiple seed values per
        method.
    :param avg_key: This is typically the seed.
    :param group_key: These are the different lines.
    :param smooth_factor: Can specify a different smooth factor per method if desired.
    :param y_bounds: What the data plot values are clipped to.
    :param y_disp_bounds: What the plotting is stopped at.
    :param ax: If not specified, one is automatically created, with the specified dimensions under `ax_dims`
    :param group_colors: If not specified defaults to `method_idxs`.
    :param num_marker_points: Key maps method name to the number of markers
        drawn on the line, NOT the number of points that are plotted! By
        default this is 8.
    :param legend: Whether to include a legend within the plot.
    :param marker_order: The marker symbols to use.
    :param marker_size: The size of the markers.
    :param override_colors: Override the color of a group to a certain color.

    :returns: The plotted figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=ax_dims)
    if rename_map is None:
        rename_map = {}
    if line_styles is None:
        line_styles = {}
    if num_marker_points is None:
        num_marker_points = {}

    if method_idxs is None:
        method_idxs = {k: i for i, k in enumerate(plot_df[group_key].unique())}

    plot_df = plot_df.copy()
    if tight:
        plt.tight_layout(pad=2.2)

    if override_colors is None:
        if group_colors is None:
            group_colors = method_idxs

        colors = sns.color_palette()
        group_colors = {k: colors[i] for k, i in group_colors.items()}
    else:
        group_colors = override_colors

    avg_y_df = plot_df.groupby([group_key, x_name]).mean()
    std_y_df = plot_df.groupby([group_key, x_name]).std()

    if y_name in plot_df.columns and y_name not in avg_y_df.columns:
        raise ValueError(
            f"Desired column {y_name} lost in the grouping. Make sure it is a numeric type"
        )

    method_runs = plot_df.groupby(group_key)[avg_key].unique()
    if fetch_std:
        y_std = y_name + "_std"
        new_df = []
        for k, sub_df in plot_df.groupby([group_key]):
            where_matches = avg_y_df.index.get_level_values(0) == k

            use_df = avg_y_df[where_matches]
            if np.isnan(sub_df.iloc[0][y_std]):
                use_df["std"] = std_y_df[where_matches][y_name]
            else:
                use_df["std"] = avg_y_df[where_matches][y_std]
            new_df.append(use_df)
        avg_y_df = pd.concat(new_df)
    else:
        avg_y_df["std"] = std_y_df[y_name]

    lines = []
    names = []

    # Update the legend info with any previously plotted lines
    if ax.get_legend() is not None:
        all_lines = ax.get_lines()

        for i, n in enumerate(ax.get_legend().get_texts()):
            names.append(n.get_text())
            lines.append((all_lines[i * 2 + 1], all_lines[i * 2]))

    if not isinstance(smooth_factor, dict):
        smooth_factor_lookup = defaultdict(lambda: smooth_factor)
    else:
        smooth_factor_lookup = defaultdict(lambda: 0.0)
        for k, v in smooth_factor.items():
            smooth_factor_lookup[k] = v

    for name, sub_df in avg_y_df.groupby(level=0):
        names.append(name)
        x_vals = sub_df.index.get_level_values(x_name).to_numpy()
        if x_vals.dtype == object:
            x_vals = np.array([rename_map.get(x, x) for x in x_vals])
        y_vals = sub_df[y_name].to_numpy()

        if x_disp_bounds is not None:
            use_y_vals = sub_df[
                sub_df.index.get_level_values(x_name) < x_disp_bounds[1]
            ][y_name].to_numpy()
        else:
            use_y_vals = y_vals
        print(
            f"{name}: n_seeds: {len(method_runs[name])} (from WB run IDs {list(method_runs[name])})",
            max(use_y_vals),
            use_y_vals[-1],
        )
        y_std = sub_df["std"].fillna(0).to_numpy()

        use_smooth_factor = smooth_factor_lookup[name]
        if use_smooth_factor != 0.0:
            y_vals = np.array(smooth_arr(y_vals, use_smooth_factor))
            y_std = np.array(smooth_arr(y_std, use_smooth_factor))

        add_kwargs = {}
        if name in line_styles:
            add_kwargs["linestyle"] = line_styles[name]
        line_to_add = ax.plot(x_vals, y_vals, **add_kwargs)
        sel_vals = [
            int(x)
            for x in np.linspace(0, len(x_vals) - 1, num=num_marker_points.get(name, 8))
        ]
        if marker_order is None:
            marker_order = MARKER_ORDER
        midx = method_idxs[name] % len(marker_order)
        ladd = ax.plot(
            x_vals[sel_vals],
            y_vals[sel_vals],
            marker_order[midx],
            label=rename_map.get(name, name),
            color=group_colors[name],
            markersize=marker_size,
        )

        lines.append((ladd[0], line_to_add[0]))

        plt.setp(line_to_add, linewidth=2, color=group_colors[name])
        min_y_fill = y_vals - y_std
        max_y_fill = y_vals + y_std

        if y_bounds is not None:
            min_y_fill = np.clip(min_y_fill, y_bounds[0], y_bounds[1])
            max_y_fill = np.clip(max_y_fill, y_bounds[0], y_bounds[1])

        ax.fill_between(
            x_vals, min_y_fill, max_y_fill, alpha=0.2, color=group_colors[name]
        )
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if x_disp_bounds is not None:
        ax.set_xlim(*x_disp_bounds)

    if xtick_fn is not None:
        plt.xticks(ax.get_xticks(), [xtick_fn(t) for t in ax.get_xticks()])
    if ytick_fn is not None:
        plt.yticks(ax.get_yticks(), [ytick_fn(t) for t in ax.get_yticks()])

    if legend:
        labs = [(i, line_to_add[0].get_label()) for i, line_to_add in enumerate(lines)]
        labs = sorted(labs, key=lambda x: method_idxs[names[x[0]]])
        kwargs = {}
        if legend_loc is not None:
            kwargs["loc"] = legend_loc
        plt.legend(
            [lines[i] for i, _ in labs],
            [x[1] for x in labs],
            fontsize=legend_font_size,
            ncol=nlegend_cols,
            **kwargs,
        )

    ax.grid(which="major", color="lightgray", linestyle="--")

    ax.set_xlabel(rename_map.get(x_name, x_name), fontsize=axes_font_size)
    ax.set_ylabel(rename_map.get(y_name, y_name), fontsize=axes_font_size)
    if x_logscale:
        ax.set_xscale("log")
    if y_logscale:
        ax.set_yscale("log")
    if title is not None and title != "":
        ax.set_title(title, fontsize=title_font_size)
    return fig, ax


def gen_fake_data(x_scale, data_key, n_runs=5):
    def create_sigmoid():
        noise = np.random.normal(0, 0.01, 100)
        x = np.linspace(0.0, 8.0, 100)
        y = 1 / (1 + np.exp(-x))
        y += noise
        return x, y

    df = None
    for i in range(n_runs):
        x, y = create_sigmoid()
        sub_df = pd.DataFrame({"_step": [int(x_i * x_scale) for x_i in x], data_key: y})
        sub_df["run"] = f"run_{i}"
        if df is None:
            df = sub_df
        else:
            df = pd.concat([df, sub_df])
    df["method"] = "fake"
    return df


def export_legend(ax, line_width, filename):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.axis("off")
    legend = ax2.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        loc="lower center",
        ncol=10,
        handlelength=2,
    )
    for line in legend.get_lines():
        line.set_linewidth(line_width)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    print("Saved legend to ", filename)


def plot_legend(
    names: List[str],
    save_path: str,
    plot_colors: Dict[str, int],
    name_map: Optional[Dict[str, str]] = None,
    linestyles: Optional[List[str]] = None,
    darkness: float = 0.1,
    marker_width: float = 0.0,
    marker_size: float = 0.0,
    line_width: float = 3.0,
    alphas: Optional[Dict[str, float]] = None,
):
    """
    :param names: The list of names to appear on the legend.
    :param plot_colors: Maps into the colors of the palette.
    :param name_map: Rename map
    """
    if name_map is None:
        name_map = {}
    if linestyles is None:
        linestyles = []
    if alphas is None:
        alphas = {}

    colors = sns.color_palette()
    group_colors = {name: colors[idx] for name, idx in plot_colors.items()}

    fig, ax = plt.subplots(figsize=(5, 4))
    for name in names:
        add_kwargs = {}
        if name in linestyles:
            linestyle = linestyles[name]
            if isinstance(linestyle, list):
                add_kwargs["linestyle"] = linestyle[0]
                add_kwargs["dashes"] = linestyle[1]
            else:
                add_kwargs["linestyle"] = linestyle

        disp_name = name_map[name]
        midx = plot_colors[name] % len(MARKER_ORDER)
        marker = MARKER_ORDER[midx]
        if marker == "x":
            marker_width = 2.0

        marker_alpha = alphas.get(name, 1.0)
        use_color = (*group_colors[name], marker_alpha)
        ax.plot(
            [0],
            [1],
            marker=marker,
            label=disp_name,
            color=use_color,
            markersize=marker_size,
            markeredgewidth=marker_width,
            # markeredgecolor=(darkness, darkness, darkness, 1),
            markeredgecolor=use_color,
            **add_kwargs,
        )
    export_legend(
        ax,
        line_width,
        save_path,
    )
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)

    query_k = "ALL_" + cfg.plot_key

    result = batch_query(
        [[query_k] for _ in cfg.methods],
        [{cfg.method_spec: v} for v in cfg.methods.values()],
        all_should_skip=[len(v) == 0 for v in cfg.methods.values()],
        all_add_info=[{"method": k} for k in cfg.methods.keys()],
        proj_cfg=OmegaConf.load(cfg.proj_cfg),
        use_cached=cfg.use_cached,
        verbose=False,
    )

    df = combine_dicts_to_df(result, query_k)
    fig = line_plot(df, "_step", cfg.plot_key, "rank", "method", **cfg.plot_params)
    fig_save("data/vis", cfg.save_name, fig)
