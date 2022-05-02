import datetime
import os
import os.path as osp
import pipes
import random
import string
import sys
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch.nn as nn
from rl_helper.common.core_utils import compress_and_filter_dict
from six.moves import shlex_quote


class Logger:
    def __init__(
        self,
        run_name: str,
        seed: int,
        log_dir: str,
        vid_dir: str,
        save_dir: str,
        smooth_len: int,
        full_cfg: Dict[str, Any],
    ):
        self.is_debug_mode = run_name == "debug"
        self._create_run_name(run_name, seed)

        self.log_dir = log_dir
        if self.log_dir != "":
            self.log_dir = osp.join(self.log_dir, self.run_name)
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.vid_dir = vid_dir
        if self.vid_dir != "":
            self.vid_dir = osp.join(self.vid_dir, self.run_name)
            if not osp.exists(self.vid_dir):
                os.makedirs(self.vid_dir)

        self.save_dir = save_dir
        if self.save_dir != "":
            self.save_dir = osp.join(self.save_dir, self.run_name)
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self._step_log_info = defaultdict(lambda: deque(maxlen=smooth_len))

        self.is_printing = True
        self.prev_steps = 0
        self.start = time.time()

    @property
    def save_path(self):
        return self.save_dir

    @property
    def vid_path(self):
        return self.vid_dir

    def disable_print(self):
        self.is_printing = False

    def collect_env_step_info(self, infos: List[Dict[str, Any]]) -> None:
        for inf in infos:
            if "episode" in inf:
                flat_inf = compress_and_filter_dict(inf)
                # Only log at the end of the episode
                for k, v in flat_inf.items():
                    self._step_log_info[k].append(v)

    def collect_infos(self, info: Dict[str, float], prefix: str = "") -> None:
        for k, v in info.items():
            self.collect_info(k, v, prefix)

    def collect_info(self, k: str, value: float, prefix: str = "") -> None:
        self._step_log_info[prefix + k].append(value)

    def _create_run_name(self, run_name, seed):
        assert run_name is not None and run_name != "", "Must specify a prefix"
        if run_name != "debug":
            d = datetime.datetime.today()
            date_id = "%i%i" % (d.month, d.day)

            chars = [
                x
                for x in string.ascii_uppercase + string.digits + string.ascii_lowercase
            ]
            rnd_id = np.random.RandomState().choice(chars, 6)
            rnd_id = "".join(rnd_id)

            self.run_name = f"{date_id}-{seed}-{rnd_id}-{run_name}"
        else:
            self.run_name = run_name
        print(f"Assigning full prefix {self.run_name}")

    def log_vals(self, key_vals, step_count):
        """
        Log key value pairs to whatever interface.
        """
        pass

    def log_args(self, args):
        pass

    def log_video(self, video_file, step_count, fps):
        pass

    def watch_model(self, model: nn.Module):
        """
        :param model: the set of parameters to watch
        """
        pass

    def interval_log(self, update_count: int, processed_env_steps: int) -> None:
        """
        Printed FPS is all inclusive of updates, evaluations, logging and everything.
        This is NOT the environment FPS.
        :param update_count: The number of updates.
        :param processed_env_steps: The number of environment samples processed.
        """
        end = time.time()

        fps = int((processed_env_steps - self.prev_steps) / (end - self.start))
        self.prev_steps = processed_env_steps
        num_eps = len(self._step_log_info.get("episode.reward", []))
        rewards = self._step_log_info.get("episode.reward", [0])

        log_dat = {}
        for k, v in self._step_log_info.items():
            log_dat[k] = np.mean(v)

        if self.is_printing:
            print("")
            print(f"Updates {update_count}, Steps {processed_env_steps}, FPS {fps}")
            print(f"Over the last {num_eps} episodes:")

            # Print log values from the updater if requested.
            for k, v in log_dat.items():
                print(f"    - {k}: {v}")

        # Log all values
        log_dat["fps"] = fps
        self.log_vals(log_dat, processed_env_steps)
        self.start = end

    def close(self):
        pass
