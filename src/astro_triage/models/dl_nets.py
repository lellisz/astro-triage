"""MLP and FT-Transformer for the deep-learning axis (ADR 0006 section 3.7, amendment A).

Both nets return raw logits (batch, 3) in CLASSES order; softmax and calibration live elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

N_CLASSES: Final = 3

MLP_CONFIG: Final[dict[str, object]] = {
    "n_layers": 3, "d_hidden": 512, "activation": "relu", "dropout": 0.1,
    "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 1024,
}
# "activation", "norm", "first_block_attention_norm" and "init" are documentation labels only:
# the code hardcodes ReGLU, prenorm, Identity first-block norm and Kaiming init.
FTT_CONFIG: Final[dict[str, object]] = {
    "n_blocks": 3, "d_token": 192, "n_heads": 8, "ffn_factor": 4 / 3, "activation": "reglu",
    "attention_dropout": 0.2, "ffn_dropout": 0.1, "residual_dropout": 0.0, "lr": 1e-4,
    "weight_decay": 1e-5, "batch_size": 1024, "norm": "prenorm",
    "first_block_attention_norm": False, "init": "kaiming",
}


def _int(config: Mapping[str, object], key: str) -> int:
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"config[{key!r}] must be int, got {type(value).__name__}")
    return value


def _float(config: Mapping[str, object], key: str) -> float:
    value = config[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"config[{key!r}] must be a number, got {type(value).__name__}")
    return float(value)


def _check_cat(n_cat: int, x_cat: Tensor | None) -> None:
    if n_cat > 0 and x_cat is None:
        raise ValueError(f"model built with n_cat={n_cat} but x_cat is None")
    if n_cat == 0 and x_cat is not None:
        raise ValueError("model built with n_cat=0 but x_cat was given; pass x_cat=None")


class MLP(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, config: Mapping[str, object]) -> None:
        super().__init__()
        self.n_cat = n_cat
        d = _int(config, "d_hidden")
        p = _float(config, "dropout")
        layers: list[nn.Module] = []
        d_in = n_cont + n_cat
        for _ in range(_int(config, "n_layers")):
            layers += [nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p)]
            d_in = d
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, N_CLASSES)

    def forward(self, x_cont: Tensor, x_cat: Tensor | None) -> Tensor:
        _check_cat(self.n_cat, x_cat)
        x = x_cont
        if self.n_cat > 0 and x_cat is not None:
            x = torch.cat([x_cont, F.one_hot(x_cat, self.n_cat).to(x_cont.dtype)], dim=1)
        return self.head(self.body(x))


class _Tokenizer(nn.Module):
    """Numeric token_j = x_j * W_j + b_j; categorical token = Embedding(code) + b_cat."""

    def __init__(self, n_cont: int, n_cat: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_cont, d))
        self.bias = nn.Parameter(torch.empty(n_cont, d))
        self.emb = nn.Embedding(n_cat, d) if n_cat > 0 else None
        self.cat_bias = nn.Parameter(torch.empty(1, d)) if n_cat > 0 else None
        # Kaiming-uniform (a=sqrt(5)) on every tokenizer tensor (gap G4, owner-visible choice).
        for t in (self.weight, self.bias, self.cat_bias, None if self.emb is None else self.emb.weight):
            if t is not None:
                nn.init.kaiming_uniform_(t, a=math.sqrt(5))

    def forward(self, x_cont: Tensor, x_cat: Tensor | None) -> Tensor:
        tokens = x_cont.unsqueeze(-1) * self.weight + self.bias
        if self.emb is not None and self.cat_bias is not None and x_cat is not None:
            tokens = torch.cat([tokens, (self.emb(x_cat) + self.cat_bias).unsqueeze(1)], dim=1)
        return tokens


class _Block(nn.Module):
    def __init__(self, d: int, config: Mapping[str, object], first: bool) -> None:
        super().__init__()
        hidden = round(d * _float(config, "ffn_factor"))
        self.hidden = hidden
        self.norm_a: nn.Module = nn.Identity() if first else nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, _int(config, "n_heads"), dropout=_float(config, "attention_dropout"), batch_first=True
        )
        self.norm_f = nn.LayerNorm(d)
        self.ffn_in = nn.Linear(d, 2 * hidden)
        self.ffn_drop = nn.Dropout(_float(config, "ffn_dropout"))
        self.ffn_out = nn.Linear(hidden, d)
        self.res_drop = nn.Dropout(_float(config, "residual_dropout"))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm_a(x)
        h = x + self.res_drop(self.attn(z, z, z, need_weights=False)[0])
        a, b = self.ffn_in(self.norm_f(h)).chunk(2, dim=-1)  # ReGLU: a * relu(b)
        return h + self.res_drop(self.ffn_out(self.ffn_drop(a * F.relu(b))))


class FTTransformer(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, config: Mapping[str, object]) -> None:
        super().__init__()
        self.n_cat = n_cat
        d = _int(config, "d_token")
        self.tok = _Tokenizer(n_cont, n_cat, d)
        self.cls = nn.Parameter(torch.empty(1, d))
        nn.init.kaiming_uniform_(self.cls, a=math.sqrt(5))
        self.blocks = nn.ModuleList(
            _Block(d, config, first=i == 0) for i in range(_int(config, "n_blocks"))
        )
        # Head on the [CLS] output: LayerNorm -> ReLU -> Linear (gap G4).
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, N_CLASSES))

    def forward(self, x_cont: Tensor, x_cat: Tensor | None) -> Tensor:
        _check_cat(self.n_cat, x_cat)
        x = self.tok(x_cont, x_cat)
        x = torch.cat([self.cls.expand(x.shape[0], 1, -1), x], dim=1)  # [CLS] first, as in rtdl (G4)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


def build_model(arch: Literal["mlp", "ftt"], n_cont: int, n_cat: int) -> nn.Module:
    if arch == "mlp":
        return MLP(n_cont, n_cat, MLP_CONFIG)
    if arch == "ftt":
        return FTTransformer(n_cont, n_cat, FTT_CONFIG)
    raise ValueError(f"unknown arch {arch!r}; expected 'mlp' or 'ftt'")


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    if not isinstance(model, FTTransformer):
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    exempt = {id(p) for p in model.tok.parameters()} | {id(model.cls)}
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            exempt |= {id(p) for p in module.parameters()}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        (no_decay if id(param) in exempt or name.endswith("bias") else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
