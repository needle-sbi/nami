from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.interpolants import (
    LinearInterpolant,
    StochasticLinearInterpolant,
    velocity_prediction,
)
from nami.interpolants.gamma import BrownianGamma, ZeroGamma
from nami.losses.regression import regression_loss
from nami.losses.stochastic_fm import stochastic_fm_loss


class ZeroField(nn.Module):
    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t, c
        return torch.zeros_like(x)


class TestStochasticFmLoss:
    def test_zero_gamma_matches_deterministic_linear_fm(self):
        """Stochastic FM with gamma=0 reduces to deterministic linear-path FM.

        ``StochasticLinearInterpolant(gamma=ZeroGamma())`` should
        produce the same loss as ``LinearInterpolant`` with Velocity
        target, because the gamma*z and gamma*z terms both vanish.  Pinned
        by exact equality.
        """
        torch.manual_seed(0)
        field = ZeroField()
        x_data = torch.randn(6, 4)
        x_noise = torch.randn(6, 4)
        t = torch.rand(6)
        z = torch.randn_like(x_data)

        deterministic = regression_loss(
            field,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            interpolant=LinearInterpolant(),
            parameterization=velocity_prediction(),
            eps_t=0.0,
            reduction="none",
        )
        stochastic = stochastic_fm_loss(
            field,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            gamma=ZeroGamma(),
            z=z,
            reduction="none",
        )

        assert torch.allclose(stochastic, deterministic, rtol=1e-6, atol=1e-6)

    def test_reductions(self):
        torch.manual_seed(0)
        field = ZeroField()
        x_data = torch.randn(5, 3)
        x_noise = torch.randn(5, 3)
        t = torch.rand(5)
        z = torch.randn_like(x_data)

        loss_none = stochastic_fm_loss(
            field,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            gamma=BrownianGamma(),
            z=z,
            reduction="none",
        )
        loss_sum = stochastic_fm_loss(
            field,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            gamma=BrownianGamma(),
            z=z,
            reduction="sum",
        )
        loss_mean = stochastic_fm_loss(
            field,
            x_data=x_data,
            x_noise=x_noise,
            t=t,
            gamma=BrownianGamma(),
            z=z,
            reduction="mean",
        )

        assert loss_none.shape == (5,)
        assert loss_sum.shape == ()
        assert loss_mean.shape == ()
        assert torch.isclose(loss_sum, loss_none.sum())
        assert torch.isclose(loss_mean, loss_none.mean())

    def test_interpolant_and_gamma_together_rejected(self):
        """`interpolant` and `gamma` are mutually exclusive constructors."""
        field = ZeroField()
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)

        with pytest.raises(ValueError, match="not both"):
            stochastic_fm_loss(
                field,
                x_data=x_data,
                x_noise=x_noise,
                interpolant=StochasticLinearInterpolant(),
                gamma=BrownianGamma(),
            )

    def test_defaults_to_brownian_stochastic_interpolant(self):
        """With neither interpolant nor gamma, the AVE default applies."""
        torch.manual_seed(0)
        field = ZeroField()
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)

        loss = stochastic_fm_loss(field, x_data=x_data, x_noise=x_noise)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_invalid_noise_shape_raises(self):
        """A wrong-shaped z fails inside the interpolant's sample step,
        which broadcasts ``gamma(t) * z`` against the deterministic mean —
        a shape mismatch surfaces as a torch RuntimeError.
        """
        field = ZeroField()
        x_data = torch.randn(4, 3)
        x_noise = torch.randn(4, 3)
        t = torch.rand(4)
        z = torch.randn(4, 2)  # wrong last dim

        with pytest.raises((ValueError, RuntimeError)):
            stochastic_fm_loss(
                field,
                x_data=x_data,
                x_noise=x_noise,
                t=t,
                gamma=BrownianGamma(),
                z=z,
            )
