"""Classifier architectures and a NumPy-facing wrapper shared by every explainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from twist.utils.serialization import read_json, write_json


class NormalizedClassifier(nn.Module):
    """Base: standardizes ``(B, T, k)`` input per channel with training-split statistics.

    Every architecture shares this preprocessing, so all explainers query the same function of
    the raw world units (plan §Week 5, task 2).
    """

    mu: torch.Tensor
    sd: torch.Tensor

    def __init__(self, k: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(k))
        self.register_buffer("sd", torch.ones(k))

    def set_normalization(self, x: np.ndarray) -> None:
        flat = x.reshape(-1, x.shape[-1])
        self.mu.copy_(torch.as_tensor(flat.mean(0), dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6, dtype=self.sd.dtype))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sd


class RecurrentClassifier(NormalizedClassifier):
    """GRU/LSTM; the last hidden state feeds a linear 2-class head."""

    def __init__(self, k: int, hidden: int = 32, layers: int = 1, cell: str = "gru"):
        super().__init__(k)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(k, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.normalize(x))
        return self.head(out[:, -1])


class _CausalConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.conv(nn.functional.pad(h, (self.pad, 0)))
        return torch.relu(out) + self.res(h)


class TCNClassifier(NormalizedClassifier):
    """Temporal convolutional network: causal dilated residual blocks (dilations 1, 2, 4, …),
    read out at the last timestep."""

    def __init__(self, k: int, hidden: int = 32, layers: int = 4, kernel: int = 3):
        super().__init__(k)
        blocks, c = [], k
        for i in range(layers):
            blocks.append(_CausalConvBlock(c, hidden, kernel, 2**i))
            c = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.normalize(x).transpose(1, 2))  # (B, hidden, T)
        return self.head(h[:, :, -1])


class TransformerClassifier(NormalizedClassifier):
    """Lightweight transformer encoder with learned positional embeddings and mean pooling."""

    def __init__(self, k: int, T: int, hidden: int = 32, layers: int = 2, heads: int = 4):
        super().__init__(k)
        self.embed = nn.Linear(k, hidden)
        self.pos = nn.Parameter(torch.zeros(1, T, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(self.normalize(x)) + self.pos[:, : x.shape[1]])
        return self.head(h.mean(dim=1))


FEATURES = ("mean", "std", "min", "max", "last", "slope", "late_mean")


def temporal_features(x: torch.Tensor) -> torch.Tensor:
    """Engineered per-channel features ``(B, k·7)``: mean, std, min, max, last value, OLS slope
    over time, and the mean of the last 10% of the series. Differentiable (min/max via
    subgradients), so gradient-based explainers can query the interpretable model too."""
    T = x.shape[1]
    t = torch.arange(T, dtype=x.dtype, device=x.device)
    tc = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    late = max(1, T // 10)
    feats = [
        x.mean(1),
        x.std(1),
        x.amin(1),
        x.amax(1),
        x[:, -1],
        torch.einsum("btk,t->bk", x, tc),
        x[:, -late:].mean(1),
    ]
    return torch.cat(feats, dim=1)


class FeatureLogisticClassifier(NormalizedClassifier):
    """Interpretable baseline: logistic regression on engineered temporal features.

    Fit with scikit-learn (L2-regularized) and loaded into a linear layer, so the model is an
    ordinary differentiable torch module at explanation time. ``feature_names`` documents the
    coefficient of each feature.
    """

    f_mu: torch.Tensor
    f_sd: torch.Tensor

    def __init__(self, k: int):
        super().__init__(k)
        n = k * len(FEATURES)
        self.register_buffer("f_mu", torch.zeros(n))
        self.register_buffer("f_sd", torch.ones(n))
        self.linear = nn.Linear(n, 1)
        self.k = k

    @property
    def feature_names(self) -> list[str]:
        return [f"{f}[x{c}]" for f in FEATURES for c in range(self.k)]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return (temporal_features(self.normalize(x)) - self.f_mu) / self.f_sd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(self.features(x))
        return torch.cat([torch.zeros_like(z), z], dim=1)  # logits for classes 0 / 1


class TorchClassifier:
    """Uniform interface: NumPy ``predict``/``logits`` for evaluation, torch ``forward`` for
    gradient-based explainers. All explainers see the same normalized input via this wrapper."""

    def __init__(self, model: nn.Module, spec: dict[str, Any]):
        self.model = model.eval()
        self.spec = spec

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def logits(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        x = np.asarray(x)
        single = x.ndim == 2
        xb = x[None] if single else x
        out = np.concatenate(
            [
                self.model(torch.as_tensor(xb[i : i + batch_size], dtype=self.dtype)).numpy()
                for i in range(0, len(xb), batch_size)
            ]
        )
        return out[0] if single else out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(-1)

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / "model.pt")
        write_json(d / "classifier_card.json", self.spec)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> TorchClassifier:
        d = Path(directory)
        spec = read_json(d / "classifier_card.json")
        model = build_model(spec)
        model.load_state_dict(torch.load(d / "model.pt", weights_only=True))
        return cls(model, spec)


def build_model(spec: dict[str, Any]) -> NormalizedClassifier:
    arch = spec["architecture"].lower()
    k, hidden, layers = spec["k"], spec.get("hidden", 32), spec.get("layers", 1)
    model: NormalizedClassifier
    if arch in ("gru", "lstm"):
        model = RecurrentClassifier(k, hidden, layers, cell=arch)
    elif arch == "tcn":
        model = TCNClassifier(k, hidden, layers, spec.get("kernel", 3))
    elif arch == "transformer":
        model = TransformerClassifier(k, spec["T"], hidden, layers, spec.get("heads", 4))
    elif arch == "logreg":
        model = FeatureLogisticClassifier(k)
    else:
        raise ValueError(f"unknown architecture {arch!r}")
    return model.double()
