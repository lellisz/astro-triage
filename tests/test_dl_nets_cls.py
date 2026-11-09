"""FT-Transformer [CLS] position (ADR 0006 section 17, G4): first token, head reads token 0."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from astro_triage.models import dl_nets as dn

N_CONT = 5
N_CAT = 4
BATCH = 6


def _model(n_cat: int) -> dn.FTTransformer:
    torch.manual_seed(123)
    model = dn.build_model("ftt", N_CONT, n_cat)
    assert isinstance(model, dn.FTTransformer)
    return model.eval()


def _inputs(n_cat: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(BATCH, N_CONT, generator=g)
    return x, (torch.randint(0, n_cat, (BATCH,), generator=g) if n_cat else None)


class _Spy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x
        return torch.zeros(x.shape[0], dn.N_CLASSES)


@pytest.mark.parametrize("n_cat", [0, N_CAT])
def test_block0_input_starts_with_cls(n_cat: int) -> None:
    model = _model(n_cat)
    x, cat = _inputs(n_cat)
    captured: list[torch.Tensor] = []
    model.blocks[0].register_forward_pre_hook(
        lambda _m, args: captured.append(args[0].detach())
    )
    with torch.no_grad():
        model(x, cat)
    seq = captured[0]
    assert seq.shape[1] == N_CONT + (1 if n_cat else 0) + 1
    expected = model.cls.detach().expand(BATCH, -1)
    assert torch.equal(seq[:, 0], expected)
    assert not torch.equal(seq[:, -1], expected)


@pytest.mark.parametrize("n_cat", [0, N_CAT])
def test_head_reads_token_zero(n_cat: int) -> None:
    model = _model(n_cat)
    x, cat = _inputs(n_cat)
    outs: list[torch.Tensor] = []
    model.blocks[-1].register_forward_hook(
        lambda _m, _a, out: outs.append(out.detach())
    )
    spy = _Spy()
    model.head = spy
    with torch.no_grad():
        model(x, cat)
    assert spy.seen is not None
    assert torch.equal(spy.seen, outs[0][:, 0])
    assert not torch.allclose(spy.seen, outs[0][:, -1])
    assert not torch.allclose(spy.seen, outs[0][:, 1])


@pytest.mark.parametrize("n_cat", [0, N_CAT])
def test_cls_first_equals_cls_last_without_positional_info(n_cat: int) -> None:
    model = _model(n_cat)
    x, cat = _inputs(n_cat)
    with torch.no_grad():
        got = model(x, cat)
        seq = torch.cat([model.tok(x, cat), model.cls.expand(BATCH, 1, -1)], dim=1)
        for block in model.blocks:
            seq = block(seq)
        want = model.head(seq[:, -1])
    assert torch.allclose(got, want, atol=1e-5)
