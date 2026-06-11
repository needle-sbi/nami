from __future__ import annotations

import pytest
import torch

from nami.core.specs import TensorSpec
from nami.divergence import ExactDivergence
from nami.fields.scalar_potential import ScalarPotentialField
from nami.losses.parameter_flow import parameter_flow_loss
from nami.scores import OracleScore


class _AnalyticPotential:
    """phi(x, theta) = sum(x): velocity 1, Laplacian 0.

    Exact solution of the parameter-flow PDE for the Gaussian-mean
    family p_theta = N(theta, 1): the mean shifts at unit rate, so
    d/dtheta log p = (x - theta) = -(d/dx log p) and the residual
    (x - theta) + 0 + 1 * (theta - x) vanishes identically.
    """

    spec = TensorSpec((1,))
    event_ndim = 1
    event_shape = (1,)

    def __call__(self, x, t=None, c=None):
        del t, c
        return x.sum(dim=-1)

    def velocity(self, x, theta, *, create_graph=True):
        del theta, create_graph
        return torch.ones_like(x)

    def velocity_field(self, *, create_graph=True):
        del create_graph
        return _AnalyticGradient()


class _AnalyticGradient:
    spec = TensorSpec((1,))
    event_ndim = 1
    event_shape = (1,)

    def __call__(self, x, t=None, c=None):
        del t, c
        # x * 0 keeps the output attached to x's graph so divergence
        # estimators can differentiate the (zero) Jacobian.
        return x * 0 + 1.0


def _gaussian_mean_scores():
    joint = OracleScore(lambda x, theta: x - theta)
    spatial = OracleScore(lambda x, theta: theta - x)
    return joint, spatial


def test_loss_zero_at_analytic_potential():
    joint, spatial = _gaussian_mean_scores()
    theta = torch.randn(64, 1)
    x = theta + torch.randn(64, 1)

    loss = parameter_flow_loss(
        _AnalyticPotential(),
        x=x,
        theta=theta,
        joint_score=joint,
        spatial_score=spatial,
        create_graph=False,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_loss_shape_and_reduction():
    torch.manual_seed(0)
    field = ScalarPotentialField(1, theta_dim=1, hidden=8, layers=2)
    joint, spatial = _gaussian_mean_scores()
    theta = torch.randn(16, 1)
    x = theta + torch.randn(16, 1)
    kwargs = {"x": x, "theta": theta, "joint_score": joint, "spatial_score": spatial}

    per_sample = parameter_flow_loss(field, reduction="none", **kwargs)
    summed = parameter_flow_loss(field, reduction="sum", **kwargs)
    mean = parameter_flow_loss(field, reduction="mean", **kwargs)

    assert per_sample.shape == (16,)
    torch.testing.assert_close(summed, per_sample.sum())
    torch.testing.assert_close(mean, per_sample.mean())

    with pytest.raises(ValueError, match="reduction"):
        parameter_flow_loss(field, reduction="median", **kwargs)


def test_loss_backward_reaches_params():
    # Guards the second-order graph: both the gradient and Laplacian
    # terms must carry gradients back to phi's parameters.
    torch.manual_seed(0)
    field = ScalarPotentialField(1, theta_dim=1, hidden=8, layers=2)
    joint, spatial = _gaussian_mean_scores()
    theta = torch.randn(8, 1)
    x = theta + torch.randn(8, 1)

    loss = parameter_flow_loss(
        field, x=x, theta=theta, joint_score=joint, spatial_score=spatial
    )
    loss.backward()

    # The residual sees only derivatives of phi, so phi's output-layer
    # bias (a constant shift) legitimately receives no gradient; the
    # weights that shape grad(phi) and Laplacian(phi) must.
    grads = [p.grad for p in field.parameters()]
    nonzero = [g for g in grads if g is not None and g.abs().sum() > 0]
    assert nonzero, "loss.backward() reached no parameters of phi"
    assert field.backbone.net[0].weight.grad is not None


def test_loss_explicit_estimator_matches_default():
    torch.manual_seed(0)
    field = ScalarPotentialField(1, theta_dim=1, hidden=8, layers=2)
    joint, spatial = _gaussian_mean_scores()
    theta = torch.randn(8, 1)
    x = theta + torch.randn(8, 1)
    kwargs = {"x": x, "theta": theta, "joint_score": joint, "spatial_score": spatial}

    default = parameter_flow_loss(field, **kwargs)
    explicit = parameter_flow_loss(
        field,
        divergence_estimator=ExactDivergence(create_graph=True),
        **kwargs,
    )

    torch.testing.assert_close(default, explicit)


def test_loss_rejects_non_flat_event():
    class _Event2:
        event_ndim = 2

    joint, spatial = _gaussian_mean_scores()
    with pytest.raises(ValueError, match="event_ndim == 1"):
        parameter_flow_loss(
            _Event2(),
            x=torch.randn(4, 1),
            theta=torch.randn(4, 1),
            joint_score=joint,
            spatial_score=spatial,
        )


def test_loss_rejects_joint_score_shape_mismatch():
    _, spatial = _gaussian_mean_scores()
    bad_joint = OracleScore(lambda _x, theta: theta.expand(*theta.shape[:-1], 2))
    with pytest.raises(ValueError, match="joint_score returned shape"):
        parameter_flow_loss(
            _AnalyticPotential(),
            x=torch.randn(4, 1),
            theta=torch.randn(4, 1),
            joint_score=bad_joint,
            spatial_score=spatial,
            create_graph=False,
        )


def test_loss_rejects_spatial_score_shape_mismatch():
    joint, _ = _gaussian_mean_scores()
    bad_spatial = OracleScore(lambda x, _theta: x.expand(*x.shape[:-1], 2))
    with pytest.raises(ValueError, match="spatial_score returned shape"):
        parameter_flow_loss(
            _AnalyticPotential(),
            x=torch.randn(4, 1),
            theta=torch.randn(4, 1),
            joint_score=joint,
            spatial_score=bad_spatial,
            create_graph=False,
        )


def test_loss_rejects_multi_theta():
    field = ScalarPotentialField(1, theta_dim=2, hidden=8, layers=2)
    joint, spatial = _gaussian_mean_scores()
    x = torch.randn(4, 1)
    theta = torch.randn(4, 2)

    with pytest.raises(ValueError, match=r"dim\(theta\) == 1"):
        parameter_flow_loss(
            field, x=x, theta=theta, joint_score=joint, spatial_score=spatial
        )
