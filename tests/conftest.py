"""Shared pytest fixtures for nami tests."""
from __future__ import annotations

import pytest
import torch

from nami.solvers.heun import Heun
from nami.solvers.ode import RK4


@pytest.fixture
def sample_tensor_4d():
    """Standard 4D test tensor with shape (2, 3, 4, 5)."""
    return torch.randn(2, 3, 4, 5)


@pytest.fixture
def sample_tensor_2d():
    """Standard 2D test tensor with shape (3, 5)."""
    return torch.randn(3, 5)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request):
    """Parametrised fixture to test multiple dtypes."""
    return request.param


@pytest.fixture
def heun_solver():
    """Default Heun solver for tests."""
    return Heun(steps=10)


@pytest.fixture
def rk4_solver():
    """Default RK4 solver for tests."""
    return RK4(steps=10)
