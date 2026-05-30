from __future__ import annotations

import torch

from nami import (
    BrownianBridgeInterpolant,
    GeneratorParams,
    ItoGeneratorOperator,
    LinearInterpolant,
    Parameterization,
    generator_prediction,
    regression_loss,
)


class _Field(torch.nn.Module):
    """Field that emits a precomputed function of (x, t, c)."""

    event_ndim = 1

    def __init__(self, output_fn):
        super().__init__()
        self._output_fn = output_fn

    def forward(self, x, t, c=None):
        return self._output_fn(x, t, c)


def test_perfect_linear_generator_field_gives_zero_loss():
    """A field whose output equals the conditional generator target
    produces a zero regression loss — the GM analogue of the FM
    perfect-velocity test.
    """
    op = ItoGeneratorOperator((4,))
    interpolant = LinearInterpolant()
    x_data = torch.randn(6, 4, dtype=torch.float64)
    x_noise = torch.randn(6, 4, dtype=torch.float64)
    t = torch.rand(6, dtype=torch.float64)

    # Use the interpolant's own target to construct the perfect field —
    # this is what a perfectly-trained model would emit.
    def perfect(x, _t, _c):  # noqa: ARG001
        state = interpolant.sample(x_noise, x_data, t)
        return interpolant.target(GeneratorParams(operator=op), state)

    field = _Field(perfect).to(dtype=torch.float64)
    loss = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=generator_prediction(op),
        eps_t=0.0,
    )
    assert torch.allclose(loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_perfect_brownian_generator_field_gives_zero_loss():
    """Same claim as the linear case but for the Brownian-bridge path.

    Pins that ``BrownianBridgeInterpolant.target(GeneratorParams(op))``
    is internally consistent — what the interpolant produces as the
    target is what a perfect field should emit, regardless of the
    bridge correction term.
    """
    op = ItoGeneratorOperator((3,))
    sigma = 0.5
    eps = 1e-4
    interpolant = BrownianBridgeInterpolant(sigma=sigma, eps=eps)
    x_data = torch.randn(8, 3, dtype=torch.float64)
    x_noise = torch.randn(8, 3, dtype=torch.float64)
    t = 0.05 + 0.9 * torch.rand(8, dtype=torch.float64)
    z = torch.randn(8, 3, dtype=torch.float64)

    def perfect(x, _t, _c):  # noqa: ARG001
        state = interpolant.sample(x_noise, x_data, t, noise=z)
        return interpolant.target(GeneratorParams(operator=op), state)

    field = _Field(perfect).to(dtype=torch.float64)
    loss = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=generator_prediction(op),
        z=z,
        eps_t=eps,
    )
    assert torch.allclose(loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_generator_prediction_matches_explicit_parameterization():
    """``generator_prediction(op)`` is structurally identical to the
    hand-rolled ``Parameterization(target=GeneratorParams(op),
    output_transform=op.project)`` — pinned at the loss level so future
    factory changes can't silently shift defaults.
    """
    op = ItoGeneratorOperator((3,), diffusion="diagonal")
    interpolant = LinearInterpolant()
    field = _Field(lambda x, t, c: torch.stack((x, torch.zeros_like(x)), dim=-2))  # noqa: ARG005
    x_data = torch.randn(8, 3, dtype=torch.float64)
    x_noise = torch.randn(8, 3, dtype=torch.float64)
    t = torch.rand(8, dtype=torch.float64)
    field = field.to(dtype=torch.float64)

    factory = generator_prediction(op)
    explicit = Parameterization(
        target=GeneratorParams(operator=op),
        output_transform=op.project,
    )

    loss_factory = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=factory,
        eps_t=0.0,
        reduction="none",
    )
    loss_explicit = regression_loss(
        field,
        x_data=x_data,
        x_noise=x_noise,
        t=t,
        interpolant=interpolant,
        parameterization=explicit,
        eps_t=0.0,
        reduction="none",
    )
    assert torch.allclose(loss_factory, loss_explicit, atol=1e-12, rtol=1e-12)


def test_generator_prediction_uses_unit_weighting():
    """``generator_prediction`` carries ω(t) = 1.

    GM has no canonical schedule-dependent weighting analogous to
    diffusion's sigma2 / SNR conventions; the factory pins ω=1 directly so
    that a future shift in :class:`Parameterization`'s default weighting
    cannot silently re-weight the GM objective.
    """
    op = ItoGeneratorOperator((3,), diffusion="diagonal")
    p = generator_prediction(op)
    t = torch.linspace(0.05, 0.95, 16)
    assert torch.equal(p.weighting(t), torch.ones_like(t))
