import os
import os.path as osp
from typing import Any, Dict, List

import pandas as pd

MISSING_VALUE = 0.2444


def combine_dicts_to_df(
    data: List[Dict[str, Any]], existing_df_key: str
) -> pd.DataFrame:
    all_dfs = []
    for r in data:
        use_df = r[existing_df_key]
        for k, v in r.items():
            if k == existing_df_key:
                continue
            use_df[k] = v
        all_dfs.append(use_df)
    return pd.concat(all_dfs)


def nice_fig_save(save_dir, save_name, fig, is_high_quality=True, verbose=True):
    """
    :param save_name: No file extension included in name.
    """
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if is_high_quality:
        full_path = osp.join(save_dir, f"{save_name}.pdf")
        fig.savefig(full_path, bbox_inches="tight", dpi=100)
    else:
        full_path = osp.join(save_dir, f"{save_name}.png")
        fig.savefig(full_path)

    if verbose:
        print(f"Saved to {full_path}")
