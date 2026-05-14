from __future__ import annotations

import pytest
import torch
from torch import nn

from nami import (
    RK4,
    ExactDivergence,
    FlowMatching,
    Parameterization,
    Score,
    StandardNormal,
    Velocity,
    velocity_prediction,
)
from nami.fields.base import VectorField
from nami.lazy import LazyDistribution, LazyProcess, UnconditionalDistribution


class PlainField(nn.Module):
    event_ndim = 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return torch.zeros_like(x)
        return (x * 0) + c.expand_as(x)


class BaseVectorField(VectorField):
    @property
    def event_ndim(self):
        return 1

    def forward(self, x, t, c=None):
        _ = t, c
        return x


def test_log_prob_without_divergence_path_raises_clean_error():
    process = FlowMatching(
        PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()
    x = torch.zeros(4, 2)

    with pytest.raises(
        TypeError,
        match=r"density evaluation requires either `estimator=\.\.\.` or a field implementing "
        r"`call_and_divergence\(x, t, c\)`",
    ):
        process.log_prob(x)


def test_log_prob_hides_internal_not_implemented_cause():
    process = FlowMatching(
        BaseVectorField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()
    x = torch.zeros(4, 2)

    with pytest.raises(
        TypeError,
        match=r"density evaluation requires either `estimator=\.\.\.` or a field implementing "
        r"`call_and_divergence\(x, t, c\)`",
    ) as excinfo:
        process.log_prob(x)

    assert excinfo.value.__cause__ is None


def test_conditional_log_prob_supports_plain_module_with_estimator():
    context = torch.randn(5, 1)
    process = FlowMatching(
        PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )(context)
    x = torch.zeros(5, 2)

    log_prob = process.log_prob(x, estimator=ExactDivergence(max_dim=8))

    assert log_prob.shape == (5,)
    assert torch.isfinite(log_prob).all()


def test_sample_return_logp_requires_divergence_path():
    process = FlowMatching(
        PlainField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=2),
    )()

    with pytest.raises(
        TypeError,
        match=r"density evaluation requires either `estimator=\.\.\.` or a field implementing "
        r"`call_and_divergence\(x, t, c\)`",
    ):
        process.sample((4,), return_logp=True)


def test_sample_return_logp_matches_separate_log_prob():
    process = FlowMatching(
        BaseVectorField(),
        StandardNormal(event_shape=(2,)),
        RK4(steps=64),
    )()
    estimator = ExactDivergence(max_dim=8)

    sample, log_prob = process.sample((5,), return_logp=True, estimator=estimator)
    expected = process.log_prob(sample, estimator=estimator)

    assert sample.shape == (5, 2)
    assert log_prob.shape == (5,)
    assert torch.allclose(log_prob, expected, atol=2e-3, rtol=2e-3)


class LearnableScaleField(nn.Module):
    event_ndim = 1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, c=None):
        _ = t, c
        return self.scale * x


def test_rsample_return_logp_is_differentiable():
    field = LearnableScaleField()
    process = FlowMatching(
        field,
        StandardNormal(event_shape=(2,)),
        RK4(steps=4),
    )()

    sample, log_prob = process.rsample(
        (4,),
        return_logp=True,
        estimator=ExactDivergence(max_dim=8, create_graph=True),
    )
    loss = sample.sum() - log_prob.sum()
    loss.backward()

    assert sample.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert field.scale.grad is not None
    assert torch.isfinite(field.scale.grad)


def test_lazy_process_and_lazy_distribution_are_split():
    wrapper = FlowMatching(PlainField(), StandardNormal(event_shape=(2,)), RK4(steps=2))
    base = UnconditionalDistribution(StandardNormal(event_shape=(2,)))

    assert isinstance(wrapper, LazyProcess)
    assert not isinstance(wrapper, LazyDistribution)
    assert isinstance(base, LazyDistribution)


class _LinearField(nn.Module):
    """Field with a non-trivial deterministic velocity for sampling tests."""

    event_ndim = 1

    def forward(self, x, t, c=None):  # noqa: ARG002
        return -0.5 * x + 0.1 * t.unsqueeze(-1)


def test_default_and_explicit_velocity_prediction_produce_identical_samples():
    """Backward-compat invariant: omitting ``parameterization`` yields the
    same samples as passing ``velocity_prediction()`` explicitly.

    Pins that the kwarg's default doesn't accidentally introduce a
    behavioural change for callers who never used the parameterisation
    vocabulary.
    """
    field = _LinearField()
    base = StandardNormal(event_shape=(3,))

    torch.manual_seed(0)
    legacy = FlowMatching(field, base, RK4(steps=8))().sample(sample_shape=(16,))

    torch.manual_seed(0)
    explicit = FlowMatching(
        field, base, RK4(steps=8), parameterization=velocity_prediction()
    )().sample(sample_shape=(16,))

    assert torch.allclose(legacy, explicit, atol=1e-12, rtol=1e-12)


def test_non_velocity_target_rejected():
    """FM only supports the Velocity target; Score / Epsilon / X0 raise.

    The conversion eps → velocity / score → velocity / x0 → velocity all
    require a NoiseSchedule that the FM Process does not carry.  Use
    Diffusion for those targets.
    """
    field = _LinearField()
    base = StandardNormal(event_shape=(3,))

    process = FlowMatching(
        field,
        base,
        RK4(steps=2),
        parameterization=Parameterization(target=Score()),
    )

    with pytest.raises(
        TypeError, match="FlowMatching supports only the Velocity target"
    ):
        process()


def test_non_identity_output_transform_changes_samples():
    """``output_transform`` is wired into the integration path.

    With a non-identity transform (here, a sign flip), the sampled
    trajectory must differ from the identity case — pins that the
    transform isn't being silently short-circuited inside
    ``_velocity``.
    """
    field = _LinearField()
    base = StandardNormal(event_shape=(3,))

    flipped = Parameterization(target=Velocity(), output_transform=lambda y: -y)

    torch.manual_seed(0)
    s_id = FlowMatching(
        field, base, RK4(steps=8), parameterization=velocity_prediction()
    )().sample(sample_shape=(16,))

    torch.manual_seed(0)
    s_flip = FlowMatching(field, base, RK4(steps=8), parameterization=flipped)().sample(
        sample_shape=(16,)
    )

    assert not torch.allclose(s_id, s_flip), (
        "output_transform appears not to be applied during integration"
    )


def test_log_prob_divergence_uses_transformed_field():
    """``log_prob`` must differentiate the *transformed* velocity, not
    the raw field, when ``output_transform`` is non-identity.

    For a linear field f(x, t) = -x and ``output_transform = lambda y: 2*y``,
    the effective velocity is ``v(x, t) = -2x`` and its divergence is
    ``-2d`` (where d = event_dim).  If the estimator differentiated the
    raw field instead, it would return ``-d`` — wrong by a factor of 2,
    which propagates into the integrated log-density.

    Pins the stage-4 review fix: the FM Process now wraps
    ``(field, output_transform)`` into ``_TransformedField`` before
    handing the estimator something to differentiate.
    """

    class _NegX(nn.Module):
        event_ndim = 1

        def forward(self, x, t, c=None):  # noqa: ARG002
            return -x

    base = StandardNormal(event_shape=(3,))
    field = _NegX()
    estimator = ExactDivergence()
    x = torch.randn(4, 3)
    t = torch.zeros(4)
    context = None

    # Identity-transform reference: divergence of -x is -3 (event_dim=3).
    process = FlowMatching(
        field,
        base,
        RK4(steps=2),
    )()
    _, div_id = process._call_and_divergence(x, t, context, estimator=estimator)
    assert torch.allclose(div_id, torch.full((4,), -3.0))

    # Non-identity transform y -> 2*y: effective velocity -2x, divergence -6.
    p_scaled = Parameterization(
        target=Velocity(),
        output_transform=lambda y: 2.0 * y,
    )
    process_scaled = FlowMatching(
        field,
        base,
        RK4(steps=2),
        parameterization=p_scaled,
    )()
    _, div_scaled = process_scaled._call_and_divergence(
        x, t, context, estimator=estimator
    )
    assert torch.allclose(div_scaled, torch.full((4,), -6.0)), (
        "divergence estimator was differentiating the raw field instead "
        "of output_transform(field(...)) — log_prob is broken for "
        "non-identity transforms"
    )


def test_log_prob_call_and_divergence_path_rejects_non_identity_transform():
    """The custom ``call_and_divergence`` field method returns the
    divergence of the raw output; using it with a non-identity
    ``output_transform`` would mix two different velocities in the
    density bookkeeping.  Force the user to pass an estimator instead.
    """

    class _FieldWithDivergence(nn.Module):
        event_ndim = 1

        def forward(self, x, t, c=None):  # noqa: ARG002
            return -x

        def call_and_divergence(self, x, t, c=None):  # noqa: ARG002
            return -x, torch.full((x.shape[0],), -float(x.shape[-1]))

    base = StandardNormal(event_shape=(3,))
    p = Parameterization(
        target=Velocity(),
        output_transform=lambda y: 2.0 * y,
    )
    process = FlowMatching(
        _FieldWithDivergence(),
        base,
        RK4(steps=2),
        parameterization=p,
    )()
    with pytest.raises(TypeError, match="non-identity output_transform"):
        process.log_prob(torch.randn(2, 3))


def test_legacy_string_parameterization_kwarg_does_not_exist():
    field = _LinearField()
    base = StandardNormal(event_shape=(3,))

    process = FlowMatching(
        field,
        base,
        RK4(steps=2),
        parameterization="velocity",  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="Parameterization"):
        process()
