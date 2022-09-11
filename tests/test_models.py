import torch

from rl_utils.models import SimpleCNN, build_rnn_state_encoder


def test_simple_cnn():
    model = SimpleCNN([64, 64, 3], 128)
    model(torch.randn(128, 3, 64, 64))

    obs_shape = {"rgb": [64, 64, 3], "other_state": [9]}
    model = SimpleCNN(obs_shape, 128)
    model({k: torch.randn(128, v[-1], *v[:2]) for k, v in obs_shape.items()})


def test_rnn():
    build_rnn_state_encoder(256, 128, "GRU", 2)

    build_rnn_state_encoder(256, 128, "LSTM", 2)
