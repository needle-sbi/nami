r"""Parameter-space paths :math:`\theta: [0, 1] \to \Theta`.

Encodes the curve along which parameter-flow transports samples.

Distinct from :class:`~nami.interpolants.protocol.Interpolant`: an
Interpolant parameterises a path in distribution space (base
:math:`\to` data); a ``ParameterPath`` parameterises a path in
parameter space (:math:`\theta_0 \to \theta_1`). They compose: a
parameter-flow sample at endpoint :math:`\theta_1` is obtained by
drawing from :math:`p_{\theta_0}` and integrating the velocity along
the ``ParameterPath``.

For ``dim(Theta) = 1`` every smooth path is trivially compatible.  For
``dim(Theta) >= 2`` the path choice is not so simple. The Euclidean
line is the default but the geometric ideal is a Fisher-Rao geodesic
in :math:`(\Theta, I(\theta))`.

TODO: Fisher-Rao geodesic path implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ParameterPath(Protocol):
    r"""Protocol for parameter-space paths."""

    def theta(self, s: torch.Tensor) -> torch.Tensor:
        r"""Evaluate :math:`\theta(s)`."""
        ...

    def dtheta_ds(self, s: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the path velocity :math:`\dot\theta(s)`."""
        ...


class LinearParameterPath:
    r"""Euclidean line: :math:`\theta(s) = (1 - s)\theta_0 + s\theta_1`.

    Parameters
    ----------
    theta_0, theta_1
        Path endpoints of identical shape ``(d_theta,)`` (or any common
        shape).  ``theta(s)`` / ``dtheta_ds(s)`` broadcast a ``(lead,)``
        path parameter to ``(lead, theta_0.shape)``.
    """

    def __init__(self, theta_0: torch.Tensor, theta_1: torch.Tensor):
        if theta_0.shape != theta_1.shape:
            msg = (
                "theta_0 and theta_1 must share a shape; got "
                f"{tuple(theta_0.shape)} and {tuple(theta_1.shape)}"
            )
            raise ValueError(msg)
        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def _expand_s(self, s: torch.Tensor) -> torch.Tensor:
        return s.reshape(*s.shape, *([1] * self.theta_0.ndim))

    def theta(self, s: torch.Tensor) -> torch.Tensor:
        s_ = self._expand_s(s)
        theta_0 = self.theta_0.to(s)
        theta_1 = self.theta_1.to(s)
        return (1 - s_) * theta_0 + s_ * theta_1

    def dtheta_ds(self, s: torch.Tensor) -> torch.Tensor:
        delta = (self.theta_1 - self.theta_0).to(s)
        return delta.expand(*s.shape, *delta.shape)


class FisherRaoGeodesicPath:
    r""":math:`\theta(s)` as the exp-map geodesic in :math:`(\Theta, I(\theta))`.

    information-geometric path. [STUB]
    """
