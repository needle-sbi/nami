from __future__ import annotations

import pytest
import torch
from torch import nn

from nami.distributions.normal import StandardNormal
from nami.interpolants import CosineInterpolant, LinearInterpolant, velocity_prediction
from nami.losses.regression import regression_loss
from nami.masking import _expand_mask, masked_fm_loss, masked_sample
from nami.solvers.ode import RK4


# helpers for this
class SimpleSetField(nn.Module):
    """Minimal MLP velocity field for (N, D) event data."""

    def __init__(self, n_objects: int, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_objects * dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_objects * dim),
        )
        self._n_objects = n_objects
        self._dim = dim

    @property
    def event_ndim(self) -> int:
        return 2

    def forward(self, x, t, c=None):
        _ = c
        flat = x.reshape(*x.shape[:-2], -1)
        t_exp = t.unsqueeze(-1).expand(*flat.shape[:-1], 1)
        out = self.net(torch.cat([flat, t_exp], dim=-1))
        return out.reshape_as(x)


class ZeroField(nn.Module):
    """Always returns zero velocity.  Useful for exact loss arithmetic."""

    @property
    def event_ndim(self) -> int:
        return 2

    def forward(self, x, t, c=None):
        _ = t, c
        return torch.zeros_like(x)


class ConstantDecayField(nn.Module):
    """Returns ``-x``, driving the state toward the origin."""

    @property
    def event_ndim(self) -> int:
        return 2

    def forward(self, x, t, c=None):
        _ = t, c
        return -x


class Scalar1DField(nn.Module):
    """event_ndim = 1  (invalid for masked_fm_loss)."""

    @property
    def event_ndim(self) -> int:
        return 1

    def forward(self, x, t, c=None):
        _ = t, c
        return torch.zeros_like(x)


# _expand_mask
# ---------------------------------------------------------------------------


class TestExpandMask:
    def test_basic_2d_event(self):
        """mask (B, N) + x (B, N, D) -> (B, N, 1)."""
        mask = torch.ones(4, 10)
        x = torch.randn(4, 10, 3)
        result = _expand_mask(mask, x, event_ndim=2)
        assert result.shape == (4, 10, 1)

    def test_with_sample_shape(self):
        """mask (B, N) + x (S, B, N, D) -> (S, B, N, 1)."""
        mask = torch.ones(4, 10)
        x = torch.randn(8, 4, 10, 3)
        result = _expand_mask(mask, x, event_ndim=2)
        assert result.shape == (8, 4, 10, 1)

    def test_3d_event(self):
        """mask (B, N) + x (B, N, H, W) -> (B, N, 1, 1)."""
        mask = torch.ones(4, 10)
        x = torch.randn(4, 10, 5, 5)
        result = _expand_mask(mask, x, event_ndim=3)
        assert result.shape == (4, 10, 1, 1)

    def test_no_batch(self):
        """mask (N,) + x (N, D) -> (N, 1)."""
        mask = torch.ones(10)
        x = torch.randn(10, 3)
        result = _expand_mask(mask, x, event_ndim=2)
        assert result.shape == (10, 1)

    def test_values_broadcast(self):
        """Verify mask values survive expansion."""
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
        x = torch.randn(2, 4, 3)
        result = _expand_mask(mask, x, event_ndim=2)
        assert result[0, 0, 0] == 1.0
        assert result[0, 2, 0] == 0.0
        assert result[1, 0, 0] == 1.0
        assert result[1, 1, 0] == 0.0

    def test_multi_sample_dims(self):
        """mask (B, N) + x (S1, S2, B, N, D) -> (S1, S2, B, N, 1)."""
        mask = torch.ones(4, 6)
        x = torch.randn(2, 3, 4, 6, 5)
        result = _expand_mask(mask, x, event_ndim=2)
        assert result.shape == (2, 3, 4, 6, 1)


# masked_fm_loss
# ---------------------------------------------------------------------------


class TestMaskedFmLoss:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        n_objects, dim, batch = 10, 4, 8
        field = SimpleSetField(n_objects, dim)
        x_target = torch.randn(batch, n_objects, dim)
        x_source = torch.randn(batch, n_objects, dim)
        mask = torch.ones(batch, n_objects)
        mask[:, 7:] = 0  # last 3 objects are padding
        return field, x_target, x_source, mask

    def test_output_is_scalar(self, setup):
        field, x_target, x_source, mask = setup
        loss = masked_fm_loss(field, x_target, x_source, mask)
        assert loss.shape == ()

    def test_reduction_none(self, setup):
        field, x_target, x_source, mask = setup
        loss = masked_fm_loss(field, x_target, x_source, mask, reduction="none")
        assert loss.shape == (8,)

    def test_reduction_sum(self, setup):
        field, x_target, x_source, mask = setup
        loss = masked_fm_loss(field, x_target, x_source, mask, reduction="sum")
        assert loss.shape == ()

    def test_all_ones_mask_matches_regression_loss(self):
        """With an all-ones mask, ``masked_fm_loss`` equals plain
        ``regression_loss`` with the same ``LinearInterpolant +
        Velocity`` setup.  Pins that the masking discipline reduces
        cleanly to the unmasked case.
        """
        torch.manual_seed(0)
        n_objects, dim, batch = 5, 3, 4
        field = SimpleSetField(n_objects, dim)
        x_target = torch.randn(batch, n_objects, dim)
        x_source = torch.randn(batch, n_objects, dim)
        mask = torch.ones(batch, n_objects)
        t = torch.rand(batch)

        loss_masked = masked_fm_loss(field, x_target, x_source, mask, t=t)
        loss_plain = regression_loss(
            field,
            x_target,
            x_source,
            t=t,
            interpolant=LinearInterpolant(),
            parameterization=velocity_prediction(),
            eps_t=0.0,
        )
        assert torch.allclose(loss_masked, loss_plain, atol=1e-6)

    def test_exact_value_with_zero_field(self):
        """With ZeroField, verify loss arithmetic against a hand computation.

        Setup:
            x_target = [[1, 2], [3, 4], [5, 6]]  (3 objects, 2 features)
            x_source = zeros
            mask     = [1, 1, 0]

        LinearInterpolant gives ut = x_source - x_target, vt = 0.
        sq_err = x_target^2.  Masked per-object means: [2.5, 12.5, 0].
        Loss = (2.5 + 12.5) / 2 = 7.5
        """
        field = ZeroField()
        x_target = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        x_source = torch.zeros_like(x_target)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        t = torch.tensor([0.5])

        loss = masked_fm_loss(field, x_target, x_source, mask, t=t, reduction="none")
        assert torch.allclose(loss, torch.tensor([7.5]))

    def test_zero_mask_gives_zero_loss(self):
        """If all objects are padded, loss should be zero."""
        torch.manual_seed(0)
        n_objects, dim, batch = 5, 3, 4
        field = SimpleSetField(n_objects, dim)
        x_target = torch.randn(batch, n_objects, dim)
        x_source = torch.randn(batch, n_objects, dim)
        mask = torch.zeros(batch, n_objects)

        loss = masked_fm_loss(field, x_target, x_source, mask)
        assert loss.item() == 0.0

    def test_event_ndim_1_raises(self):
        field = Scalar1DField()
        x = torch.randn(4, 5)
        mask = torch.ones(4)
        with pytest.raises(ValueError, match="event_ndim >= 2"):
            masked_fm_loss(field, x, x, mask)

    def test_no_event_ndim_raises(self):
        field = nn.Linear(3, 3)  # no event_ndim attribute
        x = torch.randn(4, 3)
        mask = torch.ones(4)
        with pytest.raises(ValueError, match="event_ndim is required"):
            masked_fm_loss(field, x, x, mask)

    def test_custom_interpolant(self, setup):
        field, x_target, x_source, mask = setup
        loss = masked_fm_loss(
            field, x_target, x_source, mask, interpolant=CosineInterpolant()
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_invalid_reduction(self, setup):
        field, x_target, x_source, mask = setup
        with pytest.raises(ValueError, match="reduction"):
            masked_fm_loss(field, x_target, x_source, mask, reduction="invalid")

    def test_fewer_real_objects_lower_or_equal_loss(self):
        """Masking more objects should not increase loss when padded values
        are large (because they are excluded, not averaged in)."""
        torch.manual_seed(0)
        field = ZeroField()
        x_source = torch.zeros(1, 6, 2)
        # Real objects have small values, padded have huge values
        x_target = torch.zeros(1, 6, 2)
        x_target[0, :3] = 0.1  # small
        x_target[0, 3:] = 100.0  # huge padding

        mask_all = torch.ones(1, 6)
        mask_real = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        t = torch.tensor([0.5])

        loss_all = masked_fm_loss(field, x_target, x_source, mask_all, t=t)
        loss_real = masked_fm_loss(field, x_target, x_source, mask_real, t=t)
        assert loss_real < loss_all

    def test_integer_mask(self, setup):
        """Integer (long) mask should work via .float() conversion."""
        field, x_target, x_source, _ = setup
        mask_int = torch.ones(8, 10, dtype=torch.long)
        mask_int[:, 7:] = 0
        loss = masked_fm_loss(field, x_target, x_source, mask_int)
        assert loss.shape == ()


# masked_sample
# ---------------------------------------------------------------------------


class TestMaskedSample:
    def test_output_shape(self):
        n_objects, dim, batch = 10, 4, 8
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=2)
        mask = torch.ones(batch, n_objects)
        mask[:, 7:] = 0

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))
        assert result.shape == (batch, n_objects, dim)

    def test_padded_positions_are_zero(self):
        """Padded positions should remain exactly zero after integration."""
        torch.manual_seed(42)
        n_objects, dim, batch = 8, 3, 4
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=5)
        mask = torch.ones(batch, n_objects)
        mask[:, 5:] = 0

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))
        assert torch.all(result[:, 5:] == 0.0)

    def test_real_positions_nonzero(self):
        """Real positions should generally be nonzero."""
        torch.manual_seed(42)
        n_objects, dim, batch = 8, 3, 4
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=5)
        mask = torch.ones(batch, n_objects)
        mask[:, 5:] = 0

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))
        assert not torch.all(result[:, :5] == 0.0)

    def test_all_ones_mask(self):
        """With all-ones mask, every position should be nonzero."""
        torch.manual_seed(42)
        n_objects, dim, batch = 6, 3, 4
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=5)
        mask = torch.ones(batch, n_objects)

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))
        assert result.shape == (batch, n_objects, dim)
        assert not torch.all(result == 0.0)

    def test_no_event_ndim_raises(self):
        field = nn.Linear(3, 3)
        base = StandardNormal(event_shape=(5, 3))
        solver = RK4(steps=2)
        mask = torch.ones(4, 5)
        with pytest.raises(ValueError, match="event_ndim is required"):
            masked_sample(field, base, solver, mask, sample_shape=(4,))

    def test_multi_sample_shape(self):
        """Multi-dimensional sample_shape should work."""
        n_objects, dim = 6, 3
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=2)
        mask = torch.ones(n_objects)  # no batch dim

        result = masked_sample(field, base, solver, mask, sample_shape=(2, 3))
        assert result.shape == (2, 3, n_objects, dim)

    def test_variable_mask_per_event(self):
        """Different events have different numbers of real particles."""
        torch.manual_seed(0)
        n_objects, dim, batch = 10, 3, 4
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=5)

        mask = torch.zeros(batch, n_objects)
        mask[0, :3] = 1  # event 0: 3 particles
        mask[1, :7] = 1  # event 1: 7 particles
        mask[2, :1] = 1  # event 2: 1 particle
        mask[3, :10] = 1  # event 3: all 10 particles

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))

        assert torch.all(result[0, 3:] == 0.0), "event 0 padding not zero"
        assert torch.all(result[1, 7:] == 0.0), "event 1 padding not zero"
        assert torch.all(result[2, 1:] == 0.0), "event 2 padding not zero"
        # event 3 has no padding — nothing to check

    def test_integer_mask(self):
        """Integer mask should work via .float() conversion."""
        n_objects, dim, batch = 6, 3, 4
        field = ConstantDecayField()
        base = StandardNormal(event_shape=(n_objects, dim))
        solver = RK4(steps=2)
        mask = torch.ones(batch, n_objects, dtype=torch.long)
        mask[:, 4:] = 0

        result = masked_sample(field, base, solver, mask, sample_shape=(batch,))
        assert result.shape == (batch, n_objects, dim)
        assert torch.all(result[:, 4:] == 0.0)
