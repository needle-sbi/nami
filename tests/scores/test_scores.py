from __future__ import annotations

import pytest
import torch

from nami.paths import LinearParameterPath
from nami.scores import (
    CTSMJointScore,
    DSMSpatialScore,
    MinedJointScore,
    OracleScore,
    ScoreEstimator,
)


def test_oracle_wraps_callable():
    oracle = OracleScore(lambda x, theta: x * theta)
    x = torch.randn(4, 1)
    theta = torch.randn(4, 1)
    torch.testing.assert_close(oracle(x, theta), x * theta)


def test_oracle_satisfies_protocol():
    assert isinstance(OracleScore(lambda x, _theta: x), ScoreEstimator)


def test_mined_joint_delegates():
    calls = []

    def simulator_score(x, theta):
        calls.append((x, theta))
        return torch.zeros_like(theta)

    mined = MinedJointScore(simulator_score)
    x = torch.randn(3, 2)
    theta = torch.randn(3, 1)
    out = mined(x, theta)

    assert isinstance(mined, ScoreEstimator)
    assert out.shape == (3, 1)
    assert len(calls) == 1


def test_dsm_spatial_wraps_net_and_satisfies_protocol():
    dsm = DSMSpatialScore(lambda x, theta: theta - x)
    assert isinstance(dsm, ScoreEstimator)
    x = torch.randn(5, 2)
    theta = torch.randn(5, 2)
    torch.testing.assert_close(dsm(x, theta), theta - x)


def test_ctsm_satisfies_protocol():
    path = LinearParameterPath(torch.tensor([0.0]), torch.tensor([1.0]))
    ctsm = CTSMJointScore(lambda x, _s: x.sum(-1, keepdim=True), path)
    assert isinstance(ctsm, ScoreEstimator)


def test_ctsm_dim1_chain_rule():
    path = LinearParameterPath(torch.tensor([0.0]), torch.tensor([2.0]))
    delta = 2.0

    def time_net(x, s):
        theta = path.theta(s)
        return delta * (x - theta)

    ctsm = CTSMJointScore(time_net, path)
    theta = torch.tensor([[0.5], [1.0], [1.5]])
    x = torch.randn(3, 1)
    out = ctsm(x, theta)
    assert out.shape == (3, 1)
    torch.testing.assert_close(out, x - theta)


def test_ctsm_recovers_s_from_theta_on_path():
    # A net that echoes its s argument surfaces the path inversion through
    # __call__: feeding theta(s_true) back must recover s_true.
    path = LinearParameterPath(torch.tensor([1.0, -1.0]), torch.tensor([3.0, 1.0]))
    s_true = torch.tensor([0.0, 0.25, 0.5, 1.0])
    theta = path.theta(s_true)
    ctsm = CTSMJointScore(lambda _x, s: s.unsqueeze(-1), path, directional=True)
    out = ctsm(torch.randn(4, 2), theta)
    torch.testing.assert_close(out.squeeze(-1), s_true)


def test_ctsm_rejects_off_segment_theta():
    path = LinearParameterPath(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    ctsm = CTSMJointScore(lambda x, _s: x, path, directional=True)
    off = torch.tensor([[0.5, -0.5]])
    with pytest.raises(ValueError, match="not on the training path"):
        ctsm(torch.randn(1, 2), off)


def test_ctsm_multitheta_raises_without_directional_flag():
    path = LinearParameterPath(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    ctsm = CTSMJointScore(lambda _x, s: torch.zeros_like(s).unsqueeze(-1), path)
    theta = path.theta(torch.tensor([0.3, 0.6]))
    with pytest.raises(ValueError, match="cannot recover the full joint score"):
        ctsm(torch.randn(2, 2), theta)


def test_ctsm_multitheta_directional_returns_scalar():
    path = LinearParameterPath(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

    def time_net(_x, s):
        return (3.0 * s).unsqueeze(-1)

    ctsm = CTSMJointScore(time_net, path, directional=True)
    s = torch.tensor([0.2, 0.7])
    theta = path.theta(s)
    out = ctsm(torch.randn(2, 2), theta)
    assert out.shape == (2, 1)
    torch.testing.assert_close(out.squeeze(-1), 3.0 * s)


def test_ctsm_dim1_accepts_flat_time_score():
    # A net returning shape (*lead,) (no trailing 1) is promoted to
    # (*lead, 1) before the chain-rule division.
    path = LinearParameterPath(torch.tensor([0.0]), torch.tensor([2.0]))
    ctsm = CTSMJointScore(lambda _x, s: 4.0 * s, path)
    s = torch.tensor([0.25, 0.5])
    theta = path.theta(s)
    out = ctsm(torch.randn(2, 1), theta)
    assert out.shape == (2, 1)
    torch.testing.assert_close(out.squeeze(-1), 4.0 * s / 2.0)


def test_ctsm_rejects_collinear_out_of_segment_theta():
    # theta = [2, 2] is collinear with the unit path [0,0]->[1,1] (recon
    # is exact), but maps to s = 2, outside the trained [0, 1] segment.
    path = LinearParameterPath(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    ctsm = CTSMJointScore(lambda _x, s: s.unsqueeze(-1), path, directional=True)
    off = torch.tensor([[2.0, 2.0]])
    with pytest.raises(ValueError, match="outside the trained segment"):
        ctsm(torch.randn(1, 2), off)


def test_ctsm_rejects_negative_out_of_segment_theta():
    path = LinearParameterPath(torch.tensor([0.0]), torch.tensor([2.0]))
    ctsm = CTSMJointScore(lambda _x, s: s, path)
    # theta = -1 -> s = -0.5, before the segment start.
    with pytest.raises(ValueError, match="outside the trained segment"):
        ctsm(torch.randn(1, 1), torch.tensor([[-1.0]]))


def test_ctsm_rejects_bad_net_output_shape():
    # A net whose last dim is neither absent nor 1 (here d=2) used to be
    # silently unsqueezed to (*lead, 2, 1); now it raises.
    path = LinearParameterPath(torch.tensor([0.0]), torch.tensor([2.0]))
    ctsm = CTSMJointScore(lambda x, _s: x.expand(*x.shape[:-1], 2), path)
    s = torch.tensor([0.25, 0.5])
    theta = path.theta(s)
    with pytest.raises(ValueError, match="time-score net returned shape"):
        ctsm(torch.randn(2, 1), theta)


def test_ctsm_degenerate_path_raises():
    path = LinearParameterPath(torch.tensor([1.0]), torch.tensor([1.0]))
    ctsm = CTSMJointScore(lambda x, _s: x, path)
    with pytest.raises(ValueError, match="degenerate path"):
        ctsm(torch.randn(2, 1), torch.ones(2, 1))


def test_ctsm_non_linear_path_raises_not_implemented():
    class _DummyPath:
        def theta(self, s):
            return s.unsqueeze(-1)

        def dtheta_ds(self, s):
            return torch.ones_like(s).unsqueeze(-1)

    ctsm = CTSMJointScore(lambda x, _s: x, _DummyPath())
    with pytest.raises(NotImplementedError, match="LinearParameterPath"):
        ctsm(torch.randn(2, 1), torch.randn(2, 1))
