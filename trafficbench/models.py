"""Shared-sensor LSTM and Graph WaveNet architecture adapted for modern PyTorch.

Graph WaveNet follows https://github.com/nnzhan/Graph-WaveNet/model.py.
Copyright (c) 2019 Zonghan Wu. See THIRD_PARTY_NOTICES.md for the MIT license.
Inputs: [batch, history, sensors, features]. Outputs: [batch, horizon, sensors].
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SensorLSTM(nn.Module):
    """One shared temporal model per sensor; no adjacency or neighboring speeds."""
    def __init__(self, horizon=12, hidden=64):
        super().__init__()
        self.recurrent = nn.LSTM(2, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        batch, steps, nodes, channels = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        output, _ = self.recurrent(sequence)
        return self.head(output[:, -1]).reshape(batch, nodes, -1).transpose(1, 2)


def transition(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    total = adjacency.sum(axis=1, keepdims=True)
    return np.divide(adjacency, total, out=np.zeros_like(adjacency), where=total != 0)


class DiffusionConvolution(nn.Module):
    def __init__(self, channels, support_count, dropout):
        super().__init__()
        self.projection = nn.Conv2d((1 + 2 * support_count) * channels, channels, (1, 1))
        self.dropout = dropout

    def forward(self, x, supports):
        states = [x]
        for support in supports:
            first = torch.einsum("bcnt,nm->bcmt", x, support)
            states.extend([first, torch.einsum("bcnt,nm->bcmt", first, support)])
        return F.dropout(self.projection(torch.cat(states, dim=1)), self.dropout, self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, adjacency, horizon=12, channels=32, dropout=0.3):
        super().__init__()
        nodes = len(adjacency)
        self.register_buffer("forward_support", torch.from_numpy(transition(adjacency)))
        self.register_buffer("reverse_support", torch.from_numpy(transition(adjacency.T)))
        self.nodevec1 = nn.Parameter(torch.randn(nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, nodes))
        self.start = nn.Conv2d(2, channels, (1, 1))
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.receptive_field = 1
        for _block in range(4):
            for dilation in (1, 2):
                self.filters.append(nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation)))
                self.gates.append(nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation)))
                self.skips.append(nn.Conv2d(channels, 256, (1, 1)))
                self.graph_convs.append(DiffusionConvolution(channels, 3, dropout))
                self.norms.append(nn.BatchNorm2d(channels))
                self.receptive_field += dilation
        self.end1 = nn.Conv2d(256, 512, (1, 1))
        self.end2 = nn.Conv2d(512, horizon, (1, 1))

    def forward(self, inputs):
        x = inputs.permute(0, 3, 2, 1)
        if x.shape[-1] < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.shape[-1], 0))
        x = self.start(x)
        adaptive = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        supports = [self.forward_support, self.reverse_support, adaptive]
        skip = None
        for filter_conv, gate_conv, skip_conv, graph_conv, norm in zip(
            self.filters, self.gates, self.skips, self.graph_convs, self.norms,
        ):
            residual = x
            x = torch.tanh(filter_conv(x)) * torch.sigmoid(gate_conv(x))
            contribution = skip_conv(x)
            skip = contribution if skip is None else skip[..., -contribution.shape[-1]:] + contribution
            x = graph_conv(x, supports)
            x = norm(x + residual[..., -x.shape[-1]:])
        return self.end2(F.relu(self.end1(F.relu(skip))))[..., -1]
