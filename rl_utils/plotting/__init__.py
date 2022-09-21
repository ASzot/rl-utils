from .auto_line import line_plot, plot_legend
from .auto_table import plot_table
from .utils import combine_dicts_to_df, fig_save
from .wb_query import batch_query, query, query_s

__all__ = [
    "plot_table",
    "query",
    "query_s",
    "batch_query",
    "line_plot",
    "plot_legend",
    "fig_save",
    "combine_dicts_to_df",
]
