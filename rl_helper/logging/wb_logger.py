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
from rl_helper.logging.base_logger import Logger
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
        full_cfg: Dict[str, Any],
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
        breakpoint()

        self.run = wandb.init(
            project=self.wb_proj_name,
            name=self.run_name,
            entity=self.wb_entity,
            group=group_id,
            config=full_cfg,
            reinit=True,
        )
        return wandb

    def log_args(self, args):
        wandb.config.update(args)

    def log_video(self, video_file, step_count, fps):
        if not self.should_log_vids:
            return
        wandb.log({"video": wandb.Video(video_file + ".mp4", fps=fps)}, step=step_count)

    def close(self):
        self.run.finish()
