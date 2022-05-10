import copy
import datetime
import os
import os.path as osp
import pipes
import random
import string
import sys
import time
from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf
from rl_helper.common import compress_dict
from rl_helper.logging.base_logger import Logger, LoggerCfgType
from six.moves import shlex_quote

try:
    import wandb
except:
    wandb = None


class WbLogger(Logger):
    """
    Logger for logging to the weights and W&B online service.
    """

    def __init__(
        self,
        wb_proj_name: str,
        wb_entity: str,
        run_name: str,
        seed: int,
        log_dir: str,
        vid_dir: str,
        save_dir: str,
        smooth_len: int,
        full_cfg: LoggerCfgType,
        **kwargs,
    ):
        if wandb is None:
            raise ImportError("Wandb is not installed")

        super().__init__(
            run_name, seed, log_dir, vid_dir, save_dir, smooth_len, full_cfg
        )
        if wb_proj_name == "" or wb_entity == "":
            raise ValueError(
                f"Must specify W&B project and entity name {wb_proj_name}, {wb_entity}"
            )

        self.wb_proj_name = wb_proj_name
        self.wb_entity = wb_entity
        self.wandb = self._create_wandb(full_cfg)

    def log_vals(self, key_vals, step_count):
        wandb.log(key_vals, step=int(step_count))

    def watch_model(self, model):
        wandb.watch(model)

    def _create_wandb(self, full_cfg):
        if self.run_name.count("-") >= 4:
            # Remove the seed and random ID info.
            parts = self.run_name.split("-")
            group_id = "-".join([*parts[:2], *parts[4:]])
        else:
            group_id = None
        if isinstance(full_cfg, DictConfig):
            full_cfg = OmegaConf.to_container(full_cfg, resolve=True)

        self.run = wandb.init(
            project=self.wb_proj_name,
            name=self.run_name,
            entity=self.wb_entity,
            group=group_id,
            config=full_cfg,
        )
        return wandb

    def collect_img(self, k: str, img_path: str, prefix: str = ""):
        self._step_log_info[prefix + k] = wandb.Image(img_path)

    def close(self):
        self.run.finish()
