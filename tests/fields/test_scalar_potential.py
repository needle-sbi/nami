from __future__ import annotations

import pytest
import torch

from nami.divergence import ExactDivergence
from nami.fields.scalar_potential import ScalarPotentialField


@pytest.fixture
def field() -> ScalarPotentialField:
    torch.manual_seed(0)
    return ScalarPotentialField(2, theta_dim=1, hidden=16, layers=2)


def test_forward_scalar_shape(field):
    x = torch.randn(5, 2)
    theta = torch.randn(5, 1)
    phi = field(x, None, theta)
    assert phi.shape == (5,)


def test_forward_requires_theta(field):
    with pytest.raises(ValueError, match="conditioning input"):
        field(torch.randn(5, 2), None, None)


def test_constructor_rejects_nonpositive_theta_dim():
    with pytest.raises(ValueError, match="theta_dim must be positive"):
        ScalarPotentialField(2, theta_dim=0)


def test_velocity_matches_finite_difference(field):
    x = torch.randn(4, 2, dtype=torch.float64)
    theta = torch.randn(4, 1, dtype=torch.float64)
    field = field.double()

    v = field.velocity(x, theta)

    eps = 1e-6
    for i in range(2):
        dx = torch.zeros_like(x)
        dx[:, i] = eps
        fd = (field(x + dx, None, theta) - field(x - dx, None, theta)) / (2 * eps)
        torch.testing.assert_close(v[:, i], fd, atol=1e-5, rtol=1e-5)


def test_velocity_is_curl_free(field):
    # v = grad(phi) has a symmetric Jacobian: d v_i / d x_j == d v_j / d x_i.
    field = field.double()
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(3, 1, dtype=torch.float64)

    v = field.velocity(x, theta)
    jac_rows = [
        torch.autograd.grad(v[:, i].sum(), x, create_graph=True)[0] for i in range(2)
    ]
    torch.testing.assert_close(jac_rows[0][:, 1], jac_rows[1][:, 0])


def test_velocity_works_inside_no_grad(field):
    x = torch.randn(4, 2)
    theta = torch.randn(4, 1)
    with torch.no_grad():
        v = field.velocity(x, theta, create_graph=False)
    assert v.shape == (4, 2)


def test_divergence_of_velocity_field_is_laplacian(field):
    # ExactDivergence applied to the gradient-field adapter computes
    # div(grad(phi)) = Laplacian(phi); cross-check against a manual
    # double-autograd trace.
    field = field.double()
    x = torch.randn(4, 2, dtype=torch.float64)
    theta = torch.randn(4, 1, dtype=torch.float64)

    lap = ExactDivergence(max_dim=8)(field.velocity_field(), x, None, theta)

    xx = x.clone().requires_grad_(True)
    phi = field(xx, None, theta)
    (grad_phi,) = torch.autograd.grad(phi.sum(), xx, create_graph=True)
    manual = sum(
        torch.autograd.grad(grad_phi[:, i].sum(), xx, create_graph=True)[0][:, i]
        for i in range(2)
    )

    torch.testing.assert_close(lap, manual.detach())


def test_velocity_field_adapter_exposes_protocol(field):
    adapter = field.velocity_field()
    assert adapter.event_ndim == 1
    assert adapter.event_shape == (2,)
    assert adapter.spec is field.spec
