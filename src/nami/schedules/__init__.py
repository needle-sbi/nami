"""Noise schedules ``(\\alpha(t), \\sigma(t))`` for diffusion-style transports.

Concrete schedules expose ``alpha``, ``sigma``, ``snr``, plus the SDE
``drift`` / ``diffusion`` coefficients consumed by reverse-time samplers.

References
----------
- Song et al., *Score-Based Generative Modeling through SDEs*, 2020
  (arXiv:2011.13456) — VP and VE schedules.
- Karras et al., *Elucidating the Design Space of Diffusion-Based
  Generative Models* (EDM), 2022 (arXiv:2206.00364).
"""

from __future__ import annotations

from nami.schedules.base import NoiseSchedule
from nami.schedules.edm import EDMSchedule
from nami.schedules.ve import VESchedule
from nami.schedules.vp import VPSchedule

__all__ = ["EDMSchedule", "NoiseSchedule", "VESchedule", "VPSchedule"]
