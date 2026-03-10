from __future__ import annotations

import torch
from torch import nn

from nami.interpolants.gamma import BrownianGamma, ScaledBrownianGamma, ZeroGamma
from nami.interpolants.transforms import (
    DriftFromVelocityScore,
    MarkovizationDriftFromVelocityScore,
    MirrorVelocityFromScore,
    ScoreFromNoise,
)
from nami.paths.bridge import BrownianBridgePath


def _expand_like_time(
    scale: torch.Tensor, target: torch.Tensor, event_ndim: int = 1
) -> torch.Tensor:
    lead_ndim = target.ndim - event_ndim
    n_prepend = lead_ndim - scale.ndim
    shape = (1,) * n_prepend + tuple(scale.shape) + (1,) * event_ndim
    return scale.reshape(shape)


class ScoreModel(nn.Module):
    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t
        if c is None:
            return 2.0 * x
        return 2.0 * x + c


class EtaModel(nn.Module):
    def __init__(self, score_model: nn.Module, gamma_schedule):
        super().__init__()
        self.score_model = score_model
        self.gamma_schedule = gamma_schedule

    @property
    def event_ndim(self) -> int | None:
        return getattr(self.score_model, "event_ndim", None)

    def forward(self, x, t, c=None):
        score = self.score_model(x, t, c)
        gamma = _expand_like_time(self.gamma_schedule.gamma(t), score)
        return gamma * score


class ConstantModel(nn.Module):
    def __init__(self, constant: float):
        super().__init__()
        self.constant = float(constant)
        self.last_c = None

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t
        self.last_c = c
        return torch.full_like(x, self.constant)


class TestScoreFromNoise:
    def test_recovers_score(self):
        score_model = ScoreModel()
        gamma = BrownianGamma()
        eta_model = EtaModel(score_model, gamma)
        wrapper = ScoreFromNoise(eta_model, gamma)

        x = torch.randn(12, 3)
        t = torch.linspace(0.1, 0.9, 12)
        c = torch.randn(12, 1)

        expected = score_model(x, t, c)
        actual = wrapper(x, t, c)

        assert wrapper.event_ndim == 1
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_endpoint_safe_division(self):
        class OnesEta(nn.Module):
            @property
            def event_ndim(self) -> int:
                return 1

            def forward(self, x, t, c=None):
                _ = t, c
                return torch.ones_like(x)

        wrapper = ScoreFromNoise(OnesEta(), BrownianGamma(), eps=1e-6)
        x = torch.randn(4, 2)
        t = torch.zeros(4)
        out = wrapper(x, t)

        assert torch.isfinite(out).all()


class TestDriftFromVelocityScore:
    def test_zero_gamma_reduces_to_velocity(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        wrapper = DriftFromVelocityScore(velocity, score, ZeroGamma())

        x = torch.randn(6, 4)
        t = torch.rand(6)
        c = torch.randn(6, 1)
        out = wrapper(x, t, c)

        assert wrapper.event_ndim == 1
        assert torch.allclose(out, torch.full_like(x, 2.0))
        assert velocity.last_c is c
        assert score.last_c is c

    def test_formula_with_broadcasting(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        gamma = ScaledBrownianGamma(scale=1.5)
        wrapper = DriftFromVelocityScore(velocity, score, gamma)

        x = torch.randn(5, 7, 2)
        t = torch.linspace(0.1, 0.9, 7)
        out = wrapper(x, t)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        expected = torch.full_like(x, 2.0) - gg * 3.0
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_bridge_probability_flow_reconstruction_with_sigma(self):
        sigma = 1.7
        path = BrownianBridgePath(sigma=sigma)
        x_target = torch.randn(8, 3)
        x_source = torch.randn(8, 3)
        t = torch.rand(8).clamp(0.05, 0.95)
        z = torch.randn_like(x_target)
        xt = path.sample_xt(x_target, x_source, t, z=z)

        flow_target = path.target_ut(x_target, x_source, t, xt=xt)
        score_target = path.score_target(x_target, x_source, t, xt=xt)

        class TensorField(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            @property
            def event_ndim(self) -> int:
                return 1

            def forward(self, x, t, c=None):
                _ = x, t, c
                return self.value

        gamma = ScaledBrownianGamma(scale=sigma**2)
        wrapper = DriftFromVelocityScore(
            TensorField(flow_target),
            TensorField(score_target),
            gamma,
        )
        out = wrapper(xt, t)

        t_expanded = t.reshape(t.shape + (1,) * (x_target.ndim - t.ndim))
        mu = (1.0 - t_expanded) * x_target + t_expanded * x_source
        coeff = (1.0 - 2.0 * t_expanded) / torch.clamp(
            t_expanded * (1.0 - t_expanded), min=path.eps
        )
        expected = (x_source - x_target) + coeff * (xt - mu)

        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)


class TestMirrorVelocityFromScore:
    def test_formula(self):
        score = ConstantModel(4.0)
        gamma = BrownianGamma()
        wrapper = MirrorVelocityFromScore(score, gamma)

        x = torch.randn(3, 5, 2)
        t = torch.linspace(0.1, 0.9, 5)
        c = torch.randn(5, 1)
        out = wrapper(x, t, c)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        expected = gg * 4.0
        assert wrapper.event_ndim == 1
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)
        assert score.last_c is c


class TestMarkovizationDriftFromVelocityScore:
    def test_formula_with_constant_diffusion2(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        gamma = ScaledBrownianGamma(scale=1.5)
        wrapper = MarkovizationDriftFromVelocityScore(
            velocity,
            score,
            gamma,
            diffusion2=4.0,
        )

        x = torch.randn(5, 7, 2)
        t = torch.linspace(0.1, 0.9, 7)
        out = wrapper(x, t)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        expected = torch.full_like(x, 2.0) + (-gg + 2.0) * 3.0
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_formula_with_callable_diffusion2(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        gamma = ScaledBrownianGamma(scale=1.5)
        diffusion2_fn = lambda t: 2.0 * t  # noqa: E731
        wrapper = MarkovizationDriftFromVelocityScore(
            velocity,
            score,
            gamma,
            diffusion2=diffusion2_fn,
        )

        x = torch.randn(5, 7, 2)
        t = torch.linspace(0.1, 0.9, 7)
        out = wrapper(x, t)

        gg = _expand_like_time(gamma.gamma_gamma_dot(t), x)
        g2 = _expand_like_time(2.0 * t, x)
        expected = torch.full_like(x, 2.0) + (-gg + 0.5 * g2) * 3.0
        assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_zero_diffusion2_matches_probability_flow(self):
        velocity = ConstantModel(2.0)
        score = ConstantModel(3.0)
        gamma = BrownianGamma()
        markov = MarkovizationDriftFromVelocityScore(
            velocity,
            score,
            gamma,
            diffusion2=0.0,
        )
        ode = DriftFromVelocityScore(velocity, score, gamma)

        x = torch.randn(4, 6)
        t = torch.rand(4).clamp(0.1, 0.9)
        c = torch.randn(4, 1)

        out_markov = markov(x, t, c)
        out_ode = ode(x, t, c)

        assert markov.event_ndim == 1
        assert torch.allclose(out_markov, out_ode, rtol=1e-6, atol=1e-6)
        assert velocity.last_c is c
        assert score.last_c is c
