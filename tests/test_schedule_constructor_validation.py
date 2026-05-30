from __future__ import annotations

import pytest

from nami.schedules.edm import EDMSchedule
from nami.schedules.ve import VESchedule
from nami.schedules.vp import VPSchedule


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        (
            EDMSchedule,
            {"sigma_min": 0.0, "sigma_max": 1.0},
            "sigma_min and sigma_max must be positive",
        ),
        (
            EDMSchedule,
            {"sigma_min": 2.0, "sigma_max": 1.0},
            "sigma_max must be > sigma_min",
        ),
        (EDMSchedule, {"rho": 0.0}, "rho must be positive"),
        (
            VESchedule,
            {"sigma_min": 0.0, "sigma_max": 1.0},
            "sigma_min and sigma_max must be positive",
        ),
        (
            VESchedule,
            {"sigma_min": 2.0, "sigma_max": 1.0},
            "sigma_max must be > sigma_min",
        ),
        (
            VPSchedule,
            {"beta_min": 0.0, "beta_max": 1.0},
            "beta_min and beta_max must be positive",
        ),
        (VPSchedule, {"beta_min": 2.0, "beta_max": 1.0}, "beta_max must be > beta_min"),
    ],
)
def test_schedule_constructor_validation_errors(factory, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)
