from __future__ import annotations

import pytest
import torch

import nami
from nami import (
    AllMask,
    CTMCField,
    CTMCGeneratorOperator,
    GeneratorMatching,
    KLDivergence,
    MaskingInterpolant,
    Parameterization,
    TauLeapingSampler,
    Velocity,
    cgm_loss,
    generator_prediction,
)
from nami.generators.base import GeneratorOperator
from nami.parameterizations import GeneratorParams

# --------------------------------------------------------------------------
# Operator


def test_ctmc_operator_layout():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    assert op.runtime_kind == "jump"
    assert op.parameter_shape == (3, 4)
    assert op.mask_index == 4
    assert op.vocab_size == 5


def test_ctmc_operator_rejects_tiny_vocab():
    with pytest.raises(ValueError, match="num_states must be at least 2"):
        CTMCGeneratorOperator(num_states=1, event_shape=(3,))


def test_ctmc_project_is_simplex():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    logits = torch.randn(5, 3, 4)
    probs = op.project(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5, 3), atol=1e-6)


def test_ctmc_default_divergence_is_kl():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    div = op.default_divergence()
    assert set(div) == {"rates"}
    assert isinstance(div["rates"], KLDivergence)


def test_ctmc_alpha_endpoints():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    t = torch.tensor([0.0, 1.0])
    assert torch.allclose(op.alpha(t), torch.tensor([1.0, 0.0]))


def test_ctmc_jump_step_final_unmasks_everything():
    """At the last increment alpha(t+dt)=0, so every still-masked coordinate is
    revealed."""
    op = CTMCGeneratorOperator(num_states=3, event_shape=(4,))
    x = torch.full((6, 4), op.mask_index)
    probs = torch.full((6, 4, 3), 1 / 3)
    out = op.jump_step(x, t=0.9, dt=0.1, params=probs)
    assert (out != op.mask_index).all()
    assert ((out >= 0) & (out < 3)).all()


def test_ctmc_jump_step_keeps_revealed_tokens_absorbing():
    op = CTMCGeneratorOperator(num_states=3, event_shape=(2,))
    x = torch.tensor([[1, op.mask_index]])
    probs = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    out = op.jump_step(x, t=0.0, dt=0.5, params=probs)
    assert out[0, 0].item() == 1  # already-revealed coordinate is unchanged


# --------------------------------------------------------------------------
# Masking interpolant


def test_masking_interpolant_endpoints():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    interp = MaskingInterpolant(op)
    x_data = torch.randint(0, 4, (8, 3))
    x_noise = torch.full_like(x_data, op.mask_index)

    fully_masked = interp.sample(x_noise, x_data, torch.zeros(8))
    assert (fully_masked.xt == op.mask_index).all()

    fully_data = interp.sample(x_noise, x_data, torch.ones(8))
    assert torch.equal(fully_data.xt, x_data)


def test_masking_interpolant_target_is_one_hot():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    interp = MaskingInterpolant(op)
    x_data = torch.randint(0, 4, (8, 3))
    state = interp.sample(torch.zeros_like(x_data), x_data, torch.rand(8))
    target = interp.target(GeneratorParams(operator=op), state)
    assert target.shape == (8, 3, 4)
    assert torch.equal(target.argmax(dim=-1), x_data)
    assert torch.allclose(target.sum(dim=-1), torch.ones(8, 3))


def test_masking_interpolant_rejects_velocity_target():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    interp = MaskingInterpolant(op)
    x_data = torch.randint(0, 4, (4, 3))
    state = interp.sample(torch.zeros_like(x_data), x_data, torch.rand(4))
    with pytest.raises(NotImplementedError, match="GeneratorParams"):
        interp.target(Velocity(), state)


def test_masking_interpolant_requires_ctmc_operator():
    with pytest.raises(TypeError, match="CTMCGeneratorOperator"):
        MaskingInterpolant(nami.ItoGeneratorOperator((3,)))


# --------------------------------------------------------------------------
# Field


def test_ctmc_field_output_shape():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    field = CTMCField(op, hidden=16, layers=2)
    x = torch.randint(0, op.vocab_size, (5, 3))
    out = field(x, torch.rand(5))
    assert out.shape == (5, 3, 4)


def test_ctmc_field_rejects_multi_axis_operator():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(2, 3))
    with pytest.raises(ValueError, match="single token axis"):
        CTMCField(op)


def test_ctmc_field_rejects_negative_condition_dim():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    with pytest.raises(ValueError, match="condition_dim must be non-negative"):
        CTMCField(op, condition_dim=-1)


def test_ctmc_field_rejects_wrong_token_count():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    field = CTMCField(op, hidden=16, layers=2)
    bad = torch.randint(0, op.vocab_size, (5, 2))  # 2 != num_tokens=3
    with pytest.raises(ValueError, match="token coordinates"):
        field(bad, torch.rand(5))


def test_ctmc_field_conditional_forward_concats_context():
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    field = CTMCField(op, condition_dim=2, hidden=16, layers=2)
    x = torch.randint(0, op.vocab_size, (5, 3))
    c = torch.randn(5, 2)
    out = field(x, torch.rand(5), c)
    assert out.shape == (5, 3, 4)


def test_ctmc_field_backbone_dtype_falls_back_to_default():
    """With a parameter-less backbone, the dtype probe yields the global default."""
    op = CTMCGeneratorOperator(num_states=4, event_shape=(3,))
    field = CTMCField(op, hidden=16, layers=2)
    field.backbone = torch.nn.Sequential()  # no parameters to probe
    assert field.backbone_dtype == torch.get_default_dtype()


# --------------------------------------------------------------------------
# Base distribution


def test_all_mask_base_samples_mask_tokens():
    base = AllMask((3,), mask_index=4)
    s = base.sample((7,))
    assert s.shape == (7, 3)
    assert s.dtype == torch.long
    assert (s == 4).all()


def test_all_mask_base_expand():
    base = AllMask((3,), mask_index=4).expand((5,))
    assert tuple(base.batch_shape) == (5,)
    assert base.sample().shape == (5, 3)


# --------------------------------------------------------------------------
# Sampler


def test_tau_leaping_rejects_nonpositive_steps():
    with pytest.raises(ValueError, match="steps must be positive"):
        TauLeapingSampler(steps=0)


def test_tau_leaping_integrate_rejects_nonpositive_step_override():
    """A negative ``steps`` override is truthy, so it bypasses the ``or`` default
    and must be rejected by ``integrate`` itself."""
    sampler = TauLeapingSampler(steps=8)
    with pytest.raises(ValueError, match="steps must be positive"):
        sampler.integrate(
            lambda *_args: None, torch.zeros(2, 3), t0=0.0, t1=1.0, steps=-1
        )


# --------------------------------------------------------------------------
# CGM loss + end-to-end


def _ctmc_setup(K=4, d=3, hidden=32):
    torch.manual_seed(0)
    op = CTMCGeneratorOperator(num_states=K, event_shape=(d,))
    field = CTMCField(op, hidden=hidden, layers=2)
    return op, field, MaskingInterpolant(op), generator_prediction(op)


def test_cgm_loss_on_ctmc_uses_kl_and_is_differentiable():
    op, field, interp, param = _ctmc_setup()
    x_data = torch.randint(0, op.num_states, (16, 3))
    x_noise = torch.full_like(x_data, op.mask_index)
    loss = cgm_loss(
        field,
        x_noise=x_noise,
        x_data=x_data,
        interpolant=interp,
        parameterization=param,
        eps_t=0.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_generator_matching_jump_sampling_smoke():
    op, field, _, param = _ctmc_setup()
    base = AllMask((3,), mask_index=op.mask_index)
    gm = GeneratorMatching(
        field,
        TauLeapingSampler(steps=20),
        parameterization=param,
        base=base,
        event_shape=(3,),
    )()
    s = gm.sample(sample_shape=(8,))
    assert s.shape == (8, 3)
    assert (s != op.mask_index).all()
    assert ((s >= 0) & (s < op.num_states)).all()


def test_generator_matching_jump_requires_jump_step_operator():
    """A jump-kind operator without jump_step fails clearly at sample time."""

    class _BareJump(GeneratorOperator):
        def __init__(self):
            super().__init__(runtime_kind="jump")

        @property
        def event_shape(self):
            return (2,)

        @property
        def parameter_shape(self):
            return (2,)

        def pack_params(self, *, drift, diffusion=None):
            _ = diffusion
            return drift

    op = _BareJump()

    class _Field(torch.nn.Module):
        event_ndim = 1

        def forward(self, x, t, c=None):
            _ = t, c
            return torch.zeros((*x.shape[:-1], 2))

    base = AllMask((2,), mask_index=0)
    gm = GeneratorMatching(
        _Field(),
        TauLeapingSampler(steps=3),
        parameterization=generator_prediction(op),
        base=base,
        event_shape=(2,),
        validate_args=False,
    )()
    with pytest.raises(NotImplementedError, match="jump_step"):
        gm.sample(sample_shape=(2,))


def test_generator_matching_jump_requires_solver_steps():
    """A jump solver without a ``steps`` count fails clearly at sample time."""

    class _NoStepSolver:
        steps = None

        def integrate(self, *_args, **_kwargs):  # pragma: no cover - never reached
            raise AssertionError

    op, field, _, param = _ctmc_setup()
    gm = GeneratorMatching(
        field,
        _NoStepSolver(),
        parameterization=param,
        base=AllMask((3,), mask_index=op.mask_index),
        event_shape=(3,),
        validate_args=False,
    )()
    with pytest.raises(ValueError, match="jump solver requires steps"):
        gm.sample(sample_shape=(2,))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cgm_ctmc_reduction(reduction):
    op, field, interp, param = _ctmc_setup()
    x_data = torch.randint(0, op.num_states, (8, 3))
    out = cgm_loss(
        field,
        x_noise=torch.full_like(x_data, op.mask_index),
        x_data=x_data,
        interpolant=interp,
        parameterization=param,
        eps_t=0.0,
        reduction=reduction,
    )
    assert out.ndim == 0


@pytest.mark.slow
def test_masking_ctmc_learns_marginal():
    """The KL-Bregman CGM loss trains the pure-jump generator to recover a
    peaked categorical marginal — the non-Euclidean end-to-end check."""
    torch.manual_seed(0)
    K, d = 5, 2
    op = CTMCGeneratorOperator(num_states=K, event_shape=(d,))
    field = CTMCField(op, hidden=128, layers=3)
    interp = MaskingInterpolant(op)
    param = generator_prediction(op)
    opt = torch.optim.Adam(field.parameters(), lr=3e-3)

    peaks = (1, 3)

    def sample_data(n):
        cols = []
        for peak in peaks:
            w = torch.full((K,), 0.05)
            w[peak] = 0.8
            cols.append(torch.multinomial(w, n, replacement=True))
        return torch.stack(cols, dim=-1)

    for _ in range(300):
        xd = sample_data(512)
        loss = cgm_loss(
            field,
            x_noise=torch.full_like(xd, op.mask_index),
            x_data=xd,
            interpolant=interp,
            parameterization=param,
            eps_t=0.0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    field.eval()
    gm = GeneratorMatching(
        field,
        TauLeapingSampler(steps=50),
        parameterization=param,
        base=AllMask((d,), mask_index=op.mask_index),
        event_shape=(d,),
    )()
    with torch.no_grad():
        s = gm.sample(sample_shape=(2000,))
    for col, peak in enumerate(peaks):
        mode = torch.bincount(s[:, col], minlength=K).argmax().item()
        assert mode == peak


def test_cgm_loss_ctmc_exports():
    assert nami.CTMCGeneratorOperator is CTMCGeneratorOperator
    assert isinstance(
        Parameterization(target=GeneratorParams(operator=_ctmc_setup()[0])),
        Parameterization,
    )
