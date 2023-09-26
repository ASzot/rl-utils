try:
    import wandb
except ImportError:
    wandb = None
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from rl_utils.common.core_utils import CacheHelper
from rl_utils.plotting.utils import MISSING_VALUE


def extract_query_key(k):
    if k.startswith("ALL_"):
        return k.split("ALL_")[1]
    return k


def batch_query(
    all_select_fields: List[List[str]],
    all_filter_fields: List[Dict[str, Any]],
    proj_cfg: Dict[str, Any],
    all_should_skip: Optional[List[bool]] = None,
    all_add_info: Optional[List[Dict[str, Any]]] = None,
    verbose=True,
    limit=None,
    use_cached=False,
    reduce_op: Optional[Callable[[List], float]] = None,
    error_ok: bool = False,
):
    """
    - all_should_skip: Whether to skip querying this value.
    """
    n_query = len(all_select_fields)
    if all_add_info is None:
        all_add_info = [None for _ in range(n_query)]
    if all_should_skip is None:
        all_should_skip = [False for _ in range(n_query)]

    data = []
    for select_fields, filter_fields, should_skip, add_info in zip(
        all_select_fields, all_filter_fields, all_should_skip, all_add_info
    ):
        r = []
        if not should_skip:
            r = query(
                select_fields,
                filter_fields,
                proj_cfg,
                verbose,
                limit,
                use_cached,
                reduce_op,
                error_ok=error_ok,
            )
        if len(r) == 0:
            r = [{k: MISSING_VALUE for k in select_fields}]
        for d in r:
            if add_info is None:
                data.append(d)
            else:
                data.append({**add_info, **d})
    return data


def query(
    select_fields: List[str],
    filter_fields: Dict[str, str],
    proj_cfg: Dict[str, Any],
    verbose=True,
    limit=None,
    use_cached=False,
    reduce_op: Optional[Callable[[List], float]] = None,
    error_ok: bool = False,
):
    """
    :param select_fields: The list of data to retrieve. If a field starts with
        "ALL_", then all the entries for this name from W&B are fetched. This gets
        the ENTIRE history. Other special keys include: "_runtime" (in
        seconds), "_timestamp".
    :param filter_fields: Key is the filter type (like group or tag) and value
        is the filter value (like the name of the group or tag to match)
    :param reduce_op: `np.mean` would take the average of the results.
    :param use_cached: Saves the results to disk so next time the same result is requested, it is loaded from disk rather than W&B.

    See README for more information.
    """

    wb_proj_name = proj_cfg["proj_name"]
    wb_entity = proj_cfg["wb_entity"]

    cache_name = f"wb_queries_{select_fields}_{filter_fields}"
    for bad_char in ["'", " ", "/", "[", "(", ")", "]"]:
        cache_name = cache_name.replace(bad_char, "")
    cache = CacheHelper(cache_name)

    if use_cached and cache.exists():
        return cache.load()
    if wandb is None:
        raise ValueError("Wandb is not installed")

    api = wandb.Api()

    query_dict = {}
    search_id = None

    for f, v in filter_fields.items():
        if f == "group":
            query_dict["group"] = v
        elif f == "tag":
            query_dict["tags"] = v
        elif f == "id":
            search_id = v
        else:
            query_dict["config." + f] = v

    def log(s):
        if verbose:
            print(s)

    if search_id is None:
        log("Querying with")
        log(query_dict)
        runs = api.runs(f"{wb_entity}/{wb_proj_name}", query_dict)
    else:
        log(f"Searching for ID {search_id}")
        runs = [api.run(f"{wb_entity}/{wb_proj_name}/{search_id}")]

    log(f"Returned {len(runs)} runs")

    ret_data = []
    for rank_i, run in enumerate(runs):
        dat = {"rank": rank_i}
        for f in select_fields:
            v = None
            if f == "last_model":
                parts = proj_cfg["ckpt_cfg_key"].split(".")
                model_path = run.config
                for k in parts:
                    model_path = model_path[k]
                if proj_cfg.get("ckpt_append_name", False):
                    model_path = osp.join(model_path, run.name)

                if not osp.exists(model_path):
                    raise ValueError(f"Could not locate model folder {model_path}")
                model_idxs = [
                    int(model_f.split("ckpt.")[1].split(".pth")[0])
                    for model_f in os.listdir(model_path)
                    if model_f.startswith("ckpt.")
                ]
                if len(model_idxs) == 0:
                    raise ValueError(f"No models found under {model_path}")
                max_idx = max(model_idxs)
                final_model_f = osp.join(model_path, f"ckpt.{max_idx}.pth")
                v = final_model_f
            elif f == "summary":
                v = dict(run.summary)
                v["status"] = str(run.state)
                # Filter out non-primitive values.
                v = {
                    k: k_v for k, k_v in v.items() if isinstance(k_v, (int, float, str))
                }

            elif f == "status":
                v = run.state
            elif f == "config":
                v = run.config
            elif f == "id":
                v = run.id
            elif f.startswith("config."):
                config_parts = f.split("config.")
                parts = config_parts[1].split(".")
                v = run.config
                for k in parts:
                    v = v[k]
            else:
                if f.startswith("ALL_"):
                    fetch_field = extract_query_key(f)
                    df = run.history(samples=15000)
                    if fetch_field not in df.columns:
                        raise ValueError(
                            f"Could not find {fetch_field} in {df.columns} for query {filter_fields}"
                        )
                    v = df[["_step", fetch_field]]
                else:
                    if f not in run.summary:
                        if error_ok:
                            continue
                        raise ValueError(
                            f"Could not find {f} in {run.summary.keys()} from run {run} with query {query_dict}"
                        )
                    v = run.summary[f]
            if v is not None:
                dat[f] = v
        if len(dat) > 0:
            ret_data.append(dat)
        if limit is not None and len(ret_data) >= limit:
            break

    cache.save(ret_data)
    if reduce_op is not None:
        reduce_data = defaultdict(list)
        for p in ret_data:
            for k, v in p.items():
                reduce_data[k].append(v)
        ret_data = {k: reduce_op(v) for k, v in reduce_data.items()}

    log(f"Got data {ret_data}")
    return ret_data


def query_s(
    query_str: str,
    proj_cfg: DictConfig,
    verbose=True,
    use_cached: bool = False,
):
    select_s, filter_s = query_str.split(" WHERE ")
    select_fields = select_s.replace(" ", "").split(",")

    parts = filter_s.split(" LIMIT ")
    filter_s = parts[0]

    limit = None
    if len(parts) > 1:
        limit = int(parts[1])

    filter_fields = filter_s.replace(" ", "").split(",")
    filter_fields = [s.split("=") for s in filter_fields]
    filter_fields = {k: v for k, v in filter_fields}

    return query(
        select_fields,
        filter_fields,
        proj_cfg,
        verbose=verbose,
        limit=limit,
        use_cached=use_cached,
    )


def fetch_data_from_cfg(
    plot_cfg_path: str,
    add_query_fields: Optional[List[str]] = None,
    error_ok: bool = False,
    method_key: str = "methods",
) -> pd.DataFrame:
    """
    See the README for how the YAML file at `plot_cfg_path` should be structured.
    """

    cfg = OmegaConf.load(plot_cfg_path)
    if add_query_fields is None:
        add_query_fields = []

    query_k = cfg.plot_key
    methods = cfg[method_key]

    result = batch_query(
        [[query_k, *add_query_fields, *cfg.get("add_query_keys", [])] for _ in methods],
        [{cfg.method_spec: v} for v in methods.values()],
        all_should_skip=[len(v) == 0 for v in methods.values()],
        all_add_info=[{"method": k} for k in methods.keys()],
        proj_cfg=OmegaConf.load(cfg.proj_cfg),
        use_cached=cfg.use_cached,
        verbose=False,
        error_ok=error_ok,
    )
    return pd.DataFrame(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--cache", action="store_true")
    args, query_args = parser.parse_known_args()
    query_args = " ".join(query_args)
    proj_cfg = OmegaConf.load(args.cfg)

    result = query_s(query_args, proj_cfg, use_cached=args.cache, verbose=False)
    result_summary = {}
    keys = list(result[0].keys())
    for k in keys:
        values = [r[k] for r in result]
        if isinstance(values[0], float):
            result_summary[f"{k} (mean, std)"] = (
                np.mean(values),
                np.std(values),
            )

    pprint(result)
    if len(result_summary) > 0:
        pprint(result_summary)
