from __future__ import annotations

import pytest
import torch

from nami.paths import FisherRaoGeodesicPath, LinearParameterPath, ParameterPath


def test_linear_path_endpoints():
    theta_0 = torch.tensor([-0.5])
    theta_1 = torch.tensor([0.5])
    path = LinearParameterPath(theta_0, theta_1)

    torch.testing.assert_close(path.theta(torch.zeros(3)), theta_0.expand(3, 1))
    torch.testing.assert_close(path.theta(torch.ones(3)), theta_1.expand(3, 1))


def test_linear_path_midpoint():
    path = LinearParameterPath(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(
        path.theta(torch.tensor([0.5])), torch.tensor([[0.5, 3.0]])
    )


def test_linear_path_dtheta_constant():
    theta_0 = torch.tensor([-1.0])
    theta_1 = torch.tensor([1.0])
    path = LinearParameterPath(theta_0, theta_1)

    s = torch.linspace(0.0, 1.0, 5)
    dtheta = path.dtheta_ds(s)

    assert dtheta.shape == (5, 1)
    torch.testing.assert_close(dtheta, (theta_1 - theta_0).expand(5, 1))


def test_linear_path_follows_s_dtype():
    # Endpoints are moved onto s's device/dtype, so a CPU-built path works
    # with batches cast elsewhere (here exercised via dtype).
    path = LinearParameterPath(torch.zeros(2), torch.ones(2))  # float32
    s = torch.rand(4, dtype=torch.float64)
    assert path.theta(s).dtype == torch.float64
    assert path.dtheta_ds(s).dtype == torch.float64


def test_linear_path_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share a shape"):
        LinearParameterPath(torch.zeros(1), torch.zeros(2))


def test_linear_path_satisfies_protocol():
    path = LinearParameterPath(torch.zeros(1), torch.ones(1))
    assert isinstance(path, ParameterPath)


def test_fisher_rao_is_stub():
    # The stub deliberately lacks the protocol methods so it cannot be
    # bound by accident.
    assert not isinstance(FisherRaoGeodesicPath(), ParameterPath)
