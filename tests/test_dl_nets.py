"""Tests for models/dl_nets.py (spec T-DL-05 S4)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from astro_triage.models import dl_nets as dn

N_CONT = 7
N_CAT = 4
BATCH = 8


def _inputs(n_cat: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(BATCH, N_CONT, generator=g)
    cat = torch.randint(0, n_cat, (BATCH,), generator=g) if n_cat else None
    return x, cat


def _ftt(n_cat: int = N_CAT) -> dn.FTTransformer:
    model = dn.build_model("ftt", N_CONT, n_cat)
    assert isinstance(model, dn.FTTransformer)
    return model


@pytest.mark.parametrize("arch", ["mlp", "ftt"])
@pytest.mark.parametrize("n_cat", [0, N_CAT])
def test_forward_shape_and_finite(arch: str, n_cat: int) -> None:
    model = dn.build_model(arch, N_CONT, n_cat)  # type: ignore[arg-type]
    x, cat = _inputs(n_cat)
    out = model(x, cat)
    assert out.shape == (BATCH, dn.N_CLASSES) == (BATCH, 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["mlp", "ftt"])
def test_missing_x_cat_raises(arch: str) -> None:
    model = dn.build_model(arch, N_CONT, N_CAT)  # type: ignore[arg-type]
    x, _ = _inputs(0)
    with pytest.raises(ValueError, match="x_cat"):
        model(x, None)


def test_unknown_arch_raises() -> None:
    with pytest.raises(ValueError, match="unknown arch"):
        dn.build_model("rnn", N_CONT, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_cat", [0, N_CAT])
def test_ftt_groups_follow_adr(n_cat: int) -> None:
    model = _ftt(n_cat)
    groups = dn.parameter_groups(model, 1e-5)
    assert len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 1e-5
    assert no_decay["weight_decay"] == 0.0
    assert any(p is model.cls for p in no_decay["params"])  # type: ignore[attr-defined]
    assert not any(p is model.cls for p in decay["params"])  # type: ignore[attr-defined]
    no_ids = {id(p) for p in no_decay["params"]}  # type: ignore[attr-defined]
    assert all(id(p) in no_ids for p in model.tok.parameters())
    for block in model.blocks:
        for m in (block.norm_a, block.norm_f):
            assert all(id(p) in no_ids for p in m.parameters())
    assert decay["params"], "decay group must not be empty"


@pytest.mark.parametrize("arch", ["mlp", "ftt"])
def test_groups_partition_parameters_exactly_once(arch: str) -> None:
    model = dn.build_model(arch, N_CONT, N_CAT)  # type: ignore[arg-type]
    groups = dn.parameter_groups(model, 1e-5)
    assert len(groups) == (1 if arch == "mlp" else 2)
    flat = [p for g in groups for p in g["params"]]  # type: ignore[attr-defined]
    assert len({id(p) for p in flat}) == len(flat)
    assert {id(p) for p in flat} == {id(p) for p in model.parameters()}
    if arch == "mlp":
        assert groups[0]["weight_decay"] == 1e-5


def test_no_decay_membership_is_exact() -> None:
    """'bias' name matching must not capture non-bias weights; decay holds no bias."""
    model = _ftt()
    decay, no_decay = dn.parameter_groups(model, 1e-5)
    names = {id(p): n for n, p in model.named_parameters()}
    ln_prefixes = tuple(n + "." for n, m in model.named_modules() if isinstance(m, nn.LayerNorm))
    assert ln_prefixes
    for p in no_decay["params"]:  # type: ignore[attr-defined]
        name = names[id(p)]
        assert (
            name.endswith("bias") or name == "cls" or name.startswith("tok.")
            or name.startswith(ln_prefixes)
        ), f"{name} wrongly exempt from decay"
    for p in decay["params"]:  # type: ignore[attr-defined]
        assert not names[id(p)].endswith("bias"), names[id(p)]
    decay_names = {names[id(p)] for p in decay["params"]}  # type: ignore[attr-defined]
    assert {"blocks.0.ffn_in.weight", "blocks.1.attn.in_proj_weight", "head.2.weight"} <= decay_names


def test_block_norms_and_reglu_hidden() -> None:
    model = _ftt()
    assert len(model.blocks) == 3
    assert isinstance(model.blocks[0].norm_a, nn.Identity)
    for i in (1, 2):
        assert isinstance(model.blocks[i].norm_a, nn.LayerNorm)
    for block in model.blocks:
        assert block.hidden == 256
        assert block.ffn_in.out_features == 2 * 256


@pytest.mark.parametrize(
    ("arch", "n_cat", "expected"),
    [("mlp", 0, 530947), ("mlp", N_CAT, 532995), ("ftt", 0, 894915), ("ftt", N_CAT, 895875)],
)
def test_param_count_fixed(arch: str, n_cat: int, expected: int) -> None:
    """Pinned for N_CONT=7, N_CAT=4 (reference counts at n_cont=10 differ)."""
    assert N_CONT == 7 and N_CAT == 4
    model = dn.build_model(arch, N_CONT, n_cat)  # type: ignore[arg-type]
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.parametrize("arch", ["mlp", "ftt"])
def test_unexpected_x_cat_raises(arch: str) -> None:
    model = dn.build_model(arch, N_CONT, 0)  # type: ignore[arg-type]
    x, _ = _inputs(0)
    with pytest.raises(ValueError, match="x_cat"):
        model(x, torch.zeros(BATCH, dtype=torch.long))


def test_cls_is_appended_last_and_read_by_head() -> None:
    model = _ftt().eval()
    x, cat = _inputs(N_CAT)
    with torch.no_grad():
        out = model(x, cat)
        h = model.tok(x, cat)
        h = torch.cat([h, model.cls.expand(BATCH, 1, -1)], dim=1)
        assert h.shape[1] == N_CONT + 1 + 1
        for block in model.blocks:
            h = block(h)
        assert torch.allclose(out, model.head(h[:, -1]), atol=1e-6)
        assert not torch.allclose(out, model.head(h[:, 0]), atol=1e-4)
        model.cls.add_(1.0)
        assert not torch.allclose(out, model(x, cat), atol=1e-4)


@pytest.mark.parametrize("arch", ["mlp", "ftt"])
def test_same_seed_same_initial_state(arch: str) -> None:
    torch.manual_seed(123)
    a = dn.build_model(arch, N_CONT, N_CAT).state_dict()  # type: ignore[arg-type]
    torch.manual_seed(123)
    b = dn.build_model(arch, N_CONT, N_CAT).state_dict()  # type: ignore[arg-type]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    torch.manual_seed(124)
    c = dn.build_model(arch, N_CONT, N_CAT).state_dict()  # type: ignore[arg-type]
    assert any(not torch.equal(a[k], c[k]) for k in a)
