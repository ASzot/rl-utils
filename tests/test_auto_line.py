import numpy as np
import pandas as pd
import pytest

from rl_utils.plotting import line_plot
from rl_utils.plotting.utils import fig_save


def test_line_plot():
    method_datas = {f"method{i}": np.linspace(0, 1.0 + i, 5) for i in range(5)}
    test_data = [
        (k, i, x, 0) for k, data in method_datas.items() for i, x in enumerate(data)
    ]
    df = pd.DataFrame(test_data, columns=["method", "steps", "value", "seed"])

    fig, ax = line_plot(df, "steps", "value", "seed", "method")
    fig_save("data/test", "test", fig)
