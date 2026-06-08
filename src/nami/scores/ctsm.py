r"""Joint score via Conditional Time Score Matching (CTSM)."""

from __future__ import annotations

import torch

from nami.paths.parameter import LinearParameterPath


class CTSMJointScore:
    r"""Joint score :math:`\partial_\theta\log p_\theta(x)` from a trained time-score net.

    Wraps a **trained** time-score network
    ``net(x, s) -> d/ds log p_{theta(s)}(x)`` (the scalar along-path
    log-density derivative; train it with
    :func:`~nami.losses.score_matching.time_score_matching_loss`) together
    with the :class:`~nami.paths.parameter.ParameterPath` it was trained
    on.  ``__call__(x, theta)`` inverts the path to recover ``s`` and then
    converts the time score to the joint score.

    The net learns the **scalar** derivative along the path,
    :math:`\tfrac{d}{ds}\log p = \dot\theta(s)\cdot\partial_\theta\log p`.
    Converting it to the joint score :math:`\partial_\theta\log p`
    (shape ``(*lead, d_theta)``) requires undoing the chain rule against
    the path tangent :math:`\dot\theta(s)`:

    - ``d_theta == 1``: :math:`\dot\theta` is a non-zero scalar, so
      :math:`\partial_\theta\log p = (\tfrac{d}{ds}\log p)/\dot\theta`,
      returned with shape ``(*lead, 1)``.  Exact.
    - ``d_theta > 1``: the **full** joint score is **not** recoverable
      from a single directional derivative — dividing by a vector is
      ill-posed; the time score constrains only the component of
      :math:`\partial_\theta\log p` along :math:`\dot\theta`.  By default
      this raises.  Pass ``directional=True`` to instead return the
      honest **directional** score
      :math:`\tfrac{d}{ds}\log p = \dot\theta\cdot\partial_\theta\log p`
      as a scalar, shape ``(*lead, 1)``. Feed it to
      :func:`~nami.losses.parameter_flow.path_pinned_parameter_flow_loss`
      with ``directional_score=True``, which consumes the already-contracted
      derivative directly.

    This mirrors the path-locked caveat of path-pinned parameter flow: a
    single along-path object cannot reconstruct the full multi-theta
    geometry.

    Parameters
    ----------
    trained_time_score_net: nn.Module
        Callable ``(x, s) -> Tensor`` of shape ``(*lead, 1)`` or
        ``(*lead,)``, estimating :math:`\tfrac{d}{ds}\log p_{\theta(s)}(x)`.
    path: ParameterPath
        The :class:`~nami.paths.parameter.ParameterPath` used in training.
        Path inversion is implemented for
        :class:`~nami.paths.parameter.LinearParameterPath`.
    directional: bool
        For ``d_theta > 1``, return the scalar directional score instead
        of raising.  Ignored for ``d_theta == 1``.
    on_segment_atol: float
        Absolute tolerance for the on-segment validation of ``theta``.
    """

    def __init__(
        self,
        trained_time_score_net,
        path,
        *,
        directional: bool = False,
        on_segment_atol: float = 1e-4,
    ):
        self.net = trained_time_score_net
        self.path = path
        self.directional = bool(directional)
        self.on_segment_atol = float(on_segment_atol)

    def _invert_path(self, theta: torch.Tensor) -> torch.Tensor:
        r"""Recover ``s`` from ``theta`` on a linear path; validate on-segment.

        For :math:`\theta(s) = (1-s)\theta_0 + s\theta_1`,
        :math:`s = \tfrac{(\theta - \theta_0)\cdot\Delta\theta}{|\Delta\theta|^2}`
        with :math:`\Delta\theta = \theta_1 - \theta_0`.  Rejects ``theta``
        that does not lie on the segment within tolerance.
        """
        if not isinstance(self.path, LinearParameterPath):
            msg = (
                "CTSMJointScore path inversion is implemented for "
                "LinearParameterPath only; got "
                f"{type(self.path).__name__}"
            )
            raise NotImplementedError(msg)

        theta_0 = self.path.theta_0.to(theta)
        delta = (self.path.theta_1 - self.path.theta_0).to(theta)
        denom = (delta * delta).sum()
        if denom <= 0:
            msg = "degenerate path: theta_0 == theta_1, cannot invert"
            raise ValueError(msg)

        s = ((theta - theta_0) * delta).sum(dim=-1) / denom
        # Validate theta lies on the segment: theta_0 + s * delta == theta.
        recon = theta_0 + s.unsqueeze(-1) * delta
        if not torch.allclose(recon, theta, atol=self.on_segment_atol):
            max_dev = (recon - theta).abs().max().item()
            msg = (
                "theta is not on the training path's segment within "
                f"on_segment_atol={self.on_segment_atol} (max deviation "
                f"{max_dev:.3e}); a path-locked CTSM score is only valid on "
                "its own path"
            )
            raise ValueError(msg)
        return s

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        s = self._invert_path(theta)
        time_score = self.net(x, s)
        if time_score.shape[-1:] != (1,):
            time_score = time_score.unsqueeze(-1)  # (*lead, 1)

        delta = (self.path.theta_1 - self.path.theta_0).to(theta)
        d_theta = delta.shape[-1]

        if d_theta == 1:
            dtheta = delta.reshape(*([1] * (theta.ndim - 1)), 1)
            return time_score / dtheta

        if self.directional:
            return time_score  # scalar directional score, shape (*lead, 1)

        msg = (
            "CTSMJointScore cannot recover the full joint score for "
            f"d_theta={d_theta} > 1: a single along-path time derivative "
            "constrains only the directional score.  Pass directional=True "
            "to return the scalar directional score d/ds log p instead."
        )
        raise ValueError(msg)
