"""RGL-Flow: proposed regime-guided liquid graph forecasting and its ablations.

CfC is supplied by ncps, not an ordinary RNN relabeled as a liquid network.
Input information is exactly the existing speed/time-of-day benchmark information.
Traffic regime priors are inductive biases, not a PDE conservation constraint.
"""

import math

import numpy as np
from ncps.torch import CfC
import torch
from torch import nn
from torch.nn import functional as F

from .models import transition


VARIANTS = ("regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior",
            "rglflow_msr")
REGIMES = ("free_flow", "congested", "transition")


class RegimeLiquidGraph(nn.Module):
    """Directed graph context -> CfC -> horizon-wise mixture of quantile experts."""

    def __init__(self, adjacency, mean, std, variant="regimeliquid", hidden=64):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown RGL-Flow variant: {variant}")
        self.variant = variant
        self.hidden = hidden
        self.use_experts = variant != "liquidgraph"
        self.use_prior = variant in ("regimeliquid", "regimegru")
        self.register_buffer("speed_mean", torch.tensor(float(mean)))
        self.register_buffer("speed_std", torch.tensor(float(std)))
        self.register_buffer("forward_support", torch.from_numpy(transition(adjacency)))
        self.register_buffer("reverse_support", torch.from_numpy(transition(adjacency.T)))
        nodes = len(adjacency)
        self.node_embeddings = nn.Parameter(torch.randn(nodes, 8) * 0.05)
        self.adaptive_destination = nn.Parameter(torch.randn(8, nodes) * 0.05)
        # z, dz, downstream z, upstream z, adaptive z, and four periodic features.
        self.input_projection = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.SiLU())
        self.node_projection = nn.Linear(8, 32, bias=False)
        if variant == "regimegru":
            self.temporal = nn.GRU(32, hidden, batch_first=True)
        else:
            self.temporal = CfC(32, hidden, return_sequences=False, batch_first=True,
                                backbone_units=64, backbone_layers=1, backbone_dropout=0.0)
        # Final local/upstream/downstream/learned context plus local slope/calendar.
        self.context_projection = nn.Sequential(
            nn.Linear(hidden * 4 + 8 + 6, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        widths = (64, 128, 96) if self.use_experts else (96,)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, width), nn.SiLU(), nn.Dropout(0.1), nn.Linear(width, 36))
            for width in widths
        ])
        for expert in self.experts:
            # Start with persistence medians, not an arbitrary free-flow mean.
            nn.init.normal_(expert[-1].weight, std=0.01)
            with torch.no_grad():
                expert[-1].bias.reshape(12, 3)[:, 0] = 0
                expert[-1].bias.reshape(12, 3)[:, 1] = -2
                expert[-1].bias.reshape(12, 3)[:, 2] = -1
        if self.use_experts:
            self.router = nn.Sequential(nn.Linear(hidden + 2, 48), nn.SiLU(), nn.Linear(48, 36))
            # The observed-regime prior weakens as the forecast horizon increases.
            self.register_buffer("prior_strength", torch.exp(-torch.arange(1, 13) / 6.0))

    def _encode(self, inputs, timespans=None):
        batch, steps, nodes, _features = inputs.shape
        z = inputs[..., 0:1]
        delta = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1)
        adaptive = torch.softmax(F.relu(self.node_embeddings @ self.adaptive_destination), dim=1)
        neighbor = lambda support, values: torch.einsum("nm,btmf->btnf", support, values)
        clock = inputs[..., 1:2] * (2 * math.pi)
        periodic = torch.cat([clock.sin(), clock.cos(), (2 * clock).sin(), (2 * clock).cos()], dim=-1)
        features = torch.cat([z, delta, neighbor(self.forward_support, z),
                              neighbor(self.reverse_support, z), neighbor(adaptive, z), periodic], dim=-1)
        encoded = self.input_projection(features) + self.node_projection(self.node_embeddings)[None, None]
        sequence = encoded.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 32)
        if self.variant == "regimegru":
            states, _ = self.temporal(sequence)
            temporal = states[:, -1]
        else:
            if timespans is not None:
                if timespans.shape != (batch, steps):
                    raise ValueError("timespans must be [batch, history] in five-minute units.")
                spans = timespans[:, None].expand(batch, nodes, steps).reshape(batch * nodes, steps)
                # ncps 1.0.1's wrapper squeezes batched time spans to [B], which
                # does not broadcast against [B, hidden]. Keep the official CfC
                # cell and explicitly preserve the singleton feature dimension.
                state = sequence.new_zeros(batch * nodes, self.hidden)
                for step in range(steps):
                    temporal, state = self.temporal.rnn_cell(sequence[:, step], state, spans[:, step:step + 1])
            else:
                temporal, _ = self.temporal(sequence)
        temporal = temporal.reshape(batch, nodes, self.hidden)
        final_neighbor = lambda support: torch.einsum("nm,bmf->bnf", support, temporal)
        context = torch.cat([temporal, final_neighbor(self.forward_support), final_neighbor(self.reverse_support),
                             final_neighbor(adaptive), self.node_embeddings[None].expand(batch, -1, -1),
                             z[:, -1], delta[:, -1], periodic[:, -1]], dim=-1)
        return self.context_projection(context), z[:, -1, :, 0], delta[:, -1, :, 0]

    def observed_prior(self, inputs):
        last = inputs[:, -1, :, 0] * self.speed_std + self.speed_mean
        recent_change = (inputs[:, -1, :, 0] - inputs[:, -4, :, 0]).abs() * self.speed_std
        congested = torch.sigmoid((40 - last) / 5)
        transition_probability = torch.sigmoid((recent_change - 8) / 3)
        free = (1 - congested) * (1 - transition_probability)
        transition_mass = (1 - congested) * transition_probability
        return torch.stack([free, congested, transition_mass], dim=-1).clamp_min(1e-6)

    def forward_details(self, inputs, timespans=None):
        context, latest_z, latest_delta = self._encode(inputs, timespans)
        batch, nodes, _hidden = context.shape
        distributions = []
        for expert in self.experts:
            raw = expert(context).reshape(batch, nodes, 12, 3).permute(0, 2, 1, 3)
            median = F.softplus((latest_z[:, None] + raw[..., 0]) * self.speed_std + self.speed_mean)
            lower = median * (1 - raw[..., 1].sigmoid())
            upper = median + F.softplus(raw[..., 2]) * self.speed_std
            distributions.append(torch.stack([lower, median, upper], dim=-1))
        experts = torch.stack(distributions, dim=-2)
        logits = None
        if self.use_experts:
            router_input = torch.cat([context, latest_z[..., None], latest_delta[..., None]], dim=-1)
            logits = self.router(router_input).reshape(batch, nodes, 12, 3).permute(0, 2, 1, 3)
            if self.use_prior:
                logits = logits + self.prior_strength[None, :, None, None] * self.observed_prior(inputs)[:, None].log()
            gates = logits.softmax(dim=-1)
        else:
            gates = torch.ones(batch, 12, nodes, 1, device=inputs.device)
        quantiles = (experts * gates[..., None]).sum(dim=-2)
        return {"point": (quantiles[..., 1] - self.speed_mean) / self.speed_std,
                "quantiles_mph": quantiles, "gates": gates, "gate_logits": logits}

    def forward(self, inputs, timespans=None):
        return self.forward_details(inputs, timespans)["point"]

    def training_outputs(self, inputs, truth_mph, valid):
        details = self.forward_details(inputs)
        quantiles = details["quantiles_mph"]
        levels = torch.tensor([0.1, 0.5, 0.9], device=inputs.device)
        errors = (truth_mph[..., None] - quantiles) / self.speed_std
        pinball = torch.maximum(levels * errors, (levels - 1) * errors).mean(dim=-1)
        auxiliary = 0.05 * torch.where(valid, pinball, 0).sum() / valid.sum().clamp_min(1)
        if self.use_experts:
            latest = inputs[:, -1, :, 0] * self.speed_std + self.speed_mean
            changing = (truth_mph - latest[:, None]).abs() >= 8
            # Targets supervise the router during training only; inference sees history only.
            labels = torch.where(truth_mph < 40, 1, torch.where(changing, 2, 0))
            classification = F.cross_entropy(details["gate_logits"].reshape(-1, 3), labels.flatten(), reduction="none").reshape_as(truth_mph)
            auxiliary = auxiliary + 0.02 * torch.where(valid, classification, 0).sum() / valid.sum().clamp_min(1)
        return details["point"], auxiliary


class RoutedMultiScaleLiquidGraph(RegimeLiquidGraph):
    """Multi-scale domain-routed liquid graph model.

    This candidate moves differentiable routing into the representation stage.
    The three paths see the same history-only information, but specialize in
    slow trends, local shocks and directed propagation before the existing
    probabilistic regime decoder is applied.  It is inspired by the
    domain-decomposition idea in DD-RNO and multi-scale temporal modeling, but
    does not add information unavailable to the benchmark at inference time.
    """

    def __init__(self, adjacency, mean, std, hidden=64):
        # Reuse the tested quantile/regime decoder and its auxiliary objectives.
        # The temporal encoder and input projection are replaced below, so the
        # original models and their parameterizations remain unchanged.
        super().__init__(adjacency, mean, std, variant="regimeliquid", hidden=hidden)
        self.variant = "rglflow_msr"
        # The base constructor creates these modules for the original shared
        # encoder. They are not part of this candidate's computation graph.
        del self.temporal
        del self.input_projection
        nodes = len(adjacency)
        self.branch_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.SiLU()),
            nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.SiLU()),
            nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.SiLU()),
        ])
        self.branch_temporal = nn.ModuleList([
            CfC(32, hidden, return_sequences=False, batch_first=True,
                backbone_units=64, backbone_layers=1, backbone_dropout=0.0)
            for _ in range(3)
        ])
        self.domain_router = nn.Sequential(
            nn.Linear(hidden * 3 + 6, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 3),
        )
        # A small fixed prior makes the domain semantics identifiable at the
        # start of training while the learned router remains responsible for
        # the final differentiable partition.
        self.register_buffer("domain_prior_strength", torch.tensor(0.35))

    @staticmethod
    def _causal_average(values, width):
        prefix = values[:, :1].expand(-1, width - 1, -1, -1)
        return torch.cat([prefix, values], dim=1).unfold(1, width, 1).mean(-1)

    def _run_branch(self, temporal, sequence, batch, nodes, timespans):
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * nodes, sequence.shape[1], 32)
        if timespans is None:
            state, _ = temporal(sequence)
        else:
            if timespans.shape != (batch, sequence.shape[1]):
                raise ValueError("timespans must be [batch, history] in five-minute units.")
            spans = timespans[:, None].expand(batch, nodes, sequence.shape[1]).reshape(batch * nodes, sequence.shape[1])
            state = sequence.new_zeros(batch * nodes, self.hidden)
            for step in range(sequence.shape[1]):
                state, _ = temporal.rnn_cell(sequence[:, step], state, spans[:, step:step + 1])
        return state.reshape(batch, nodes, self.hidden)

    def _domain_evidence(self, inputs, adaptive, z, delta):
        forward = torch.einsum("nm,btmf->btnf", self.forward_support, z)
        reverse = torch.einsum("nm,btmf->btnf", self.reverse_support, z)
        # Slow, shock and propagation evidence are all history-only.  The
        # learned router can override these soft priors when validation data
        # shows that a different decomposition is more useful.
        shock = delta.abs()[..., 0] + 0.5 * torch.cat(
            [torch.zeros_like(delta[:, :1]), delta[:, 1:] - delta[:, :-1]], dim=1
        ).abs()[..., 0]
        propagation = (forward - reverse).abs()[..., 0]
        smooth = 1.0 / (1.0 + delta.abs()[..., 0])
        return torch.stack([smooth, shock, propagation], dim=-1)

    def _encode(self, inputs, timespans=None):
        batch, steps, nodes, _features = inputs.shape
        z = inputs[..., 0:1]
        delta = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1)
        second = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:] - delta[:, :-1]], dim=1)
        adaptive = torch.softmax(F.relu(self.node_embeddings @ self.adaptive_destination), dim=1)
        neighbor = lambda support, values: torch.einsum("nm,btmf->btnf", support, values)
        slow = self._causal_average(z, width=4)
        clock = inputs[..., 1:2] * (2 * math.pi)
        periodic = torch.cat([clock.sin(), clock.cos(), (2 * clock).sin(), (2 * clock).cos()], dim=-1)

        branch_inputs = [
            torch.cat([z, slow, neighbor(self.forward_support, slow),
                       neighbor(self.reverse_support, slow), neighbor(adaptive, slow), periodic], dim=-1),
            torch.cat([delta, second, neighbor(self.forward_support, delta),
                       neighbor(self.reverse_support, delta), neighbor(adaptive, delta), z, periodic], dim=-1),
            torch.cat([z, neighbor(self.forward_support, z), neighbor(self.reverse_support, z),
                       neighbor(adaptive, z), neighbor(self.forward_support, z) -
                       neighbor(self.reverse_support, z), delta, periodic], dim=-1),
        ]
        branches = []
        for projection, temporal, values in zip(self.branch_projections, self.branch_temporal, branch_inputs):
            encoded = projection(values) + self.node_projection(self.node_embeddings)[None, None]
            branches.append(self._run_branch(temporal, encoded, batch, nodes, timespans))
        branch_states = torch.stack(branches, dim=-2)
        latest_periodic = periodic[:, -1]
        route_input = torch.cat([branch_states.reshape(batch, nodes, -1),
                                 z[:, -1], delta[:, -1], latest_periodic], dim=-1)
        logits = self.domain_router(route_input)
        evidence = self._domain_evidence(inputs, adaptive, z, delta)
        logits = logits + self.domain_prior_strength * evidence[:, -1]
        domains = logits.softmax(dim=-1)
        fused = (branch_states * domains[..., None]).sum(dim=-2)
        final_neighbor = lambda support: torch.einsum("nm,bmf->bnf", support, fused)
        context = torch.cat([fused, final_neighbor(self.forward_support),
                             final_neighbor(self.reverse_support), final_neighbor(adaptive),
                             self.node_embeddings[None].expand(batch, -1, -1),
                             z[:, -1], delta[:, -1], latest_periodic], dim=-1)
        return self.context_projection(context), z[:, -1, :, 0], delta[:, -1, :, 0], logits, domains

    def forward_details(self, inputs, timespans=None):
        context, latest_z, latest_delta, domain_logits, domains = self._encode(inputs, timespans)
        batch, nodes, _hidden = context.shape
        distributions = []
        for expert in self.experts:
            raw = expert(context).reshape(batch, nodes, 12, 3).permute(0, 2, 1, 3)
            median = F.softplus((latest_z[:, None] + raw[..., 0]) * self.speed_std + self.speed_mean)
            lower = median * (1 - raw[..., 1].sigmoid())
            upper = median + F.softplus(raw[..., 2]) * self.speed_std
            distributions.append(torch.stack([lower, median, upper], dim=-1))
        experts = torch.stack(distributions, dim=-2)
        router_input = torch.cat([context, latest_z[..., None], latest_delta[..., None]], dim=-1)
        logits = self.router(router_input).reshape(batch, nodes, 12, 3).permute(0, 2, 1, 3)
        logits = logits + self.prior_strength[None, :, None, None] * self.observed_prior(inputs)[:, None].log()
        gates = logits.softmax(dim=-1)
        quantiles = (experts * gates[..., None]).sum(dim=-2)
        return {"point": (quantiles[..., 1] - self.speed_mean) / self.speed_std,
                "quantiles_mph": quantiles, "gates": gates, "gate_logits": logits,
                "domain_gates": domains, "domain_logits": domain_logits}

    def training_outputs(self, inputs, truth_mph, valid):
        point, auxiliary = super().training_outputs(inputs, truth_mph, valid)
        details = self.forward_details(inputs)
        # Keep the representation router aligned with its history-only
        # decomposition without leaking target values into its forward path.
        z = inputs[..., 0:1]
        delta = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1)
        adaptive = torch.softmax(F.relu(self.node_embeddings @ self.adaptive_destination), dim=1)
        evidence = self._domain_evidence(inputs, adaptive, z, delta)[:, -1]
        labels = evidence.detach().argmax(dim=-1)
        route_loss = F.cross_entropy(details["domain_logits"].reshape(-1, 3), labels.reshape(-1))
        return point, auxiliary + 0.01 * route_loss


def build_proposed(data, name):
    if name == "rglflow_msr":
        return RoutedMultiScaleLiquidGraph(data.adjacency, data.mean, data.std)
    return RegimeLiquidGraph(data.adjacency, data.mean, data.std, variant=name)
