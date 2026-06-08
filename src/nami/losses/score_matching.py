r"""Training losses for the upstream score networks of parameter-flow.

The parameter-flow loss
(:func:`~nami.losses.parameter_flow.parameter_flow_loss` /
:func:`~nami.losses.parameter_flow.path_pinned_parameter_flow_loss`)
consumes two frozen score targets

- a joint score :math:`\partial_\theta\log p_\theta(x)` and
- a spatial score :math:`\nabla_x\log p_\theta(x)`.

When those are not closed-form oracles they are supplied by trained score networks
(:class:`~nami.scores.ctsm.CTSMJointScore`,
:class:`~nami.scores.dsm.DSMSpatialScore`).
This module trains the score networks.

Three losses:

- :func:`denoising_score_matching_loss`: standard Gaussian DSM (Vincent
  2011) for the **spatial** score :math:`\nabla_x\log p_\theta(x)`,
  amortised over :math:`\theta`.

- :func:`ctsm_loss`: the full Conditional Time Score Matching
  objective (Yu et al. 2025, Eq. 8): self-contained training of the
  time score :math:`\partial_t\log p_t(x)` along a Gaussian conditional
  path :math:`p_t(x|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I)`, no
  density knowledge required. Theorem 1 guarantees the minimiser is
  the marginal time score.

- :func:`time_score_matching_loss`: the general Theorem-3 regression
  against a caller-supplied per-sample target (conditional or
  marginal).

References
----------
- Yu, Klami, Hyvärinen, Korba, Chehab, *Density Ratio Estimation with
  Conditional Probability Paths*, ICML 2025 (arXiv:2502.02300).
- Choi et al., *Density Ratio Estimation via Infinitesimal
  Classification*, 2022: the integrated time-score identity
  :math:`\log[p_1(x)/p_0(x)] = \int_0^1 \partial_t\log p_t(x)\,dt`.
- Vincent, *A Connection Between Score Matching and Denoising
  Autoencoders*, 2011.
"""

from __future__ import annotations

import torch

from nami.losses._common import (
    leading_shape,
    reduce_loss,
    require_event_ndim,
    sample_t,
)


def denoising_score_matching_loss(
    net,
    *,
    x: torch.Tensor,
    theta: torch.Tensor,
    sigma: float | torch.Tensor,
    noise: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Denoising score-matching loss conditioned on theta.

    Corrupts clean samples with isotropic Gaussian noise of scale
    :math:`\sigma` and regresses the network onto the denoising target,
    which is the (negative) score of the Gaussian corruption kernel:

    .. math::

        \mathcal{L} = \mathbb{E}_{x, \tilde x}\,
            \bigl\| \mathrm{net}(\tilde x, \theta)
                    - \tfrac{x - \tilde x}{\sigma^2} \bigr\|^2,
        \qquad \tilde x = x + \sigma\,\varepsilon,\ \varepsilon\sim N(0, I).

    At the optimum :math:`\mathrm{net}(\cdot, \theta) =
    \nabla_x\log p^\sigma_\theta` (the score of the sigma-smoothed
    density); for small :math:`\sigma` this approximates
    :math:`\nabla_x\log p_\theta`.  The trained network is wrapped by
    :class:`~nami.scores.dsm.DSMSpatialScore`.

    Parameters
    ----------
    net: nn.Module
        Network ``(x_tilde, theta) -> Tensor`` returning the estimated
        spatial score, shape ``(*lead, d_x)``.
    x: torch.Tensor
        Clean samples from :math:`p_\theta`, shape ``(*lead, *event)``.
    theta: torch.Tensor
        Conditioning parameters, shape ``(*lead, d_theta)``.
    sigma: float | torch.Tensor
        Noise scale.  A python float for a single scale, or a tensor
        broadcastable to ``x`` for a per-sample range (sample it in the
        caller and pass it in).
    noise: torch.Tensor | None
        Optional pre-drawn standard-normal noise of ``x``'s shape; drawn
        internally when ``None`` (the usual path).
    reduction: str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    event_ndim = require_event_ndim(net) if hasattr(net, "event_ndim") else 1
    lead = leading_shape(x, event_ndim)

    sigma_t = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    if noise is None:
        noise = torch.randn_like(x)
    x_tilde = x + sigma_t * noise
    # the denoising target
    target = (x - x_tilde) / sigma_t.pow(2)

    pred = net(x_tilde, theta)
    if pred.shape != target.shape:
        msg = (
            f"net returned shape {tuple(pred.shape)}; expected x's shape "
            f"{tuple(target.shape)}"
        )
        raise ValueError(msg)

    per_sample = (pred - target).pow(2).reshape(*lead, -1).sum(dim=-1)
    return reduce_loss(per_sample, reduction)


def _schedule_rates(
    schedule, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate ``\alpha, \sigma, \dot\alpha, \dot\sigma`` at ``t``.

    Derivatives come from autograd through the schedule, so any
    :class:`~nami.schedules.base.NoiseSchedule` works without exposing
    derivative methods, or in other words, the CTSM construction is path-agnostic.
    """
    with torch.enable_grad():
        t_req = t.detach().requires_grad_(True)
        a = schedule.alpha(t_req)
        s = schedule.sigma(t_req)
        (a_dot,) = torch.autograd.grad(a.sum(), t_req, retain_graph=True)
        (s_dot,) = torch.autograd.grad(s.sum(), t_req)
    return a.detach(), s.detach(), a_dot, s_dot


def _conditional_time_score(
    z_flat: torch.Tensor,
    eps_flat: torch.Tensor,
    a_dot: torch.Tensor,
    s: torch.Tensor,
    s_dot: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form ``\partial_t \log p_t(x_t|z)`` on the Gaussian path.

    With :math:`x_t = \alpha_t z + \sigma_t\varepsilon` and
    :math:`p_t(x|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I)` (Yu et al.
    2025, Eq. 14), substituting :math:`x_t - \alpha_t z = \sigma_t
    \varepsilon` into :math:`\partial_t` of the Gaussian log-density
    gives (Eqs. 15-16):

    .. math::

        \partial_t \log p_t(x_t|z)
        = \frac{\dot\alpha_t}{\sigma_t}\,(\varepsilon\cdot z)
        + \frac{\dot\sigma_t}{\sigma_t}\,(\lVert\varepsilon\rVert^2 - d).
    """
    d = eps_flat.shape[-1]
    dot_ez = (eps_flat * z_flat).sum(dim=-1)
    eps_sq = eps_flat.pow(2).sum(dim=-1)
    return (a_dot / s) * dot_ez + (s_dot / s) * (eps_sq - d)


def ctsm_loss(
    net,
    *,
    x_data: torch.Tensor,
    t: torch.Tensor | None = None,
    schedule,
    eps_t: float = 1e-3,
    noise: torch.Tensor | None = None,
    weighting="normalised",
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Conditional Time Score Matching loss.

    Trains a time-score network :math:`s_\phi(x, t) \approx
    \partial_t\log p_t(x)` along the Gaussian conditional path

    .. math::

        x_t = \alpha_t z + \sigma_t\,\varepsilon,
        \qquad z \sim p_{\mathrm{data}},\ \varepsilon \sim \mathcal{N}(0, I),

    by regressing against the conditional time score
    :math:`\partial_t\log p_t(x_t|z)`, which is closed-form (see
    :func:`_conditional_time_score`). No knowledge of the marginal
    density is needed.  This is the denoising trick applied to time
    scores: by Yu et al.'s Theorem 1 the objective equals the
    intractable marginal (TSM) objective up to an additive constant, so
    the minimiser is the marginal time score
    :math:`\partial_t\log p_t(x) = \mathbb{E}[\partial_t\log p_t(x|z)
    \mid x_t = x]`.

    The trained network supports the integrated density-ratio identity
    (Choi et al. 2022) :math:`\log[p_{t_1}(x)/p_{t_0}(x)] =
    \int_{t_0}^{t_1} s_\phi(x, \tau)\,d\tau` and is the natural training
    surface for :class:`~nami.scores.ctsm.CTSMJointScore` when the
    interpolation variable is identified with a parameter path
    coordinate.  Note the identification caveat: this objective trains
    time scores along the schedule's noise path; using it for a
    parameter path requires the path marginals to admit the same
    Gaussian conditional structure.

    Parameters
    ----------
    net: nn.Module
        Time-score network ``(x, t) -> (*lead,)`` or ``(*lead, 1)``.
    x_data: torch.Tensor
        Samples ``z`` from the data distribution, shape
        ``(*lead, *event)``.  These are the conditioning variables
        (``z = x_1`` in the paper's Eq. 14).
    t: torch.Tensor | None
        Optional pre-sampled times.  When ``None``, drawn from
        ``U[eps_t, 1 - eps_t]``.
    schedule: NoiseSchedule
        :class:`~nami.schedules.base.NoiseSchedule` supplying
        ``alpha(t)`` / ``sigma(t)``; derivatives are taken by autograd,
        so any schedule works (VP is the paper's default).
    eps_t
        Endpoint protection for the automatic ``t`` draw, the target
        variance diverges where :math:`\sigma_t \to 0`.
    noise: torch.Tensor | None
        Optional pre-drawn standard-normal ``\varepsilon`` of
        ``x_data``'s shape; drawn internally when ``None``.
    weighting: str | Callable[[torch.Tensor], torch.Tensor]
        ``"normalised"`` (default) applies the time-score normalisation
        of Eq. 20, :math:`\lambda(t) \propto 1/\mathbb{E}[(\partial_t
        \log p_t(x|z))^2]`, in the closed form
        :math:`\lambda(t)^{-1} = (\dot\alpha_t/\sigma_t)^2\,
        \mathbb{E}\lVert z\rVert^2 + 2d\,(\dot\sigma_t/\sigma_t)^2`
        (cross term vanishes by symmetry; :math:`\mathbb{E}\lVert
        z\rVert^2` estimated from the batch).  ``"uniform"`` disables
        it; a callable ``t -> weight`` supplies a custom schedule.
    reduction: str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    event_ndim = require_event_ndim(net) if hasattr(net, "event_ndim") else 1
    lead = leading_shape(x_data, event_ndim)
    t = sample_t(x_data, lead, t, eps_t)

    a, s, a_dot, s_dot = _schedule_rates(schedule, t)

    if noise is None:
        noise = torch.randn_like(x_data)
    shape = (*lead, *([1] * event_ndim))
    x_t = a.reshape(shape) * x_data + s.reshape(shape) * noise

    z_flat = x_data.reshape(*lead, -1)
    eps_flat = noise.reshape(*lead, -1)
    target = _conditional_time_score(z_flat, eps_flat, a_dot, s, s_dot)

    pred = net(x_t, t)
    if pred.shape[-1:] == (1,) and pred.ndim == len(lead) + 1:
        pred = pred.squeeze(-1)
    if pred.shape != target.shape:
        msg = (
            f"net output (squeezed) shape {tuple(pred.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
        raise ValueError(msg)

    if weighting == "normalised":
        d = eps_flat.shape[-1]
        m2 = z_flat.pow(2).sum(dim=-1).mean().detach()
        lam = 1.0 / ((a_dot / s).pow(2) * m2 + 2 * d * (s_dot / s).pow(2))
    elif weighting == "uniform":
        lam = torch.ones_like(target)
    elif callable(weighting):
        lam = weighting(t)
        if lam.shape != target.shape:
            lam = lam.expand_as(target)
    else:
        msg = "weighting must be 'normalised', 'uniform', or a callable"
        raise ValueError(msg)

    per_sample = lam.detach() * (pred - target).pow(2)
    return reduce_loss(per_sample, reduction)


def time_score_matching_loss(
    net,
    *,
    x: torch.Tensor,
    s: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Time score-matching loss.

    Regresses a time-score network ``net(x, s)`` onto
    :math:`\tfrac{d}{ds}\log p_{\theta(s)}(x)`, the derivative of the
    log-density along a :class:`~nami.paths.parameter.ParameterPath`.

    A plain regression against a caller-supplied per-sample target, which
    may be either the marginal time score (analytic toys) or any
    tractable conditional whose posterior mean is the marginal. The
    theorem guarantees the two objectives share a minimiser. Use this
    when the target comes from outside (closed forms, instrumented
    simulators). For the self-contained case, training from samples
    alone along a Gaussian conditional path, no density knowledge, use
    :func:`ctsm_loss`.

    Parameters
    ----------
    net: nn.Module
        Time-score network ``(x, s) -> Tensor`` of shape ``(*lead, 1)``
        (or ``(*lead,)``; squeezed-compatible with ``target``).
    x: torch.Tensor
        Samples from :math:`p_{\theta(s)}`, shape ``(*lead, *event)``.
    s: torch.Tensor
        Path parameters, shape ``(*lead,)`` or ``(*lead, 1)``.
    target: torch.Tensor
        The per-sample regression target
        :math:`\tfrac{d}{ds}\log p_{\theta(s)}(x)`, broadcastable to the
        network output.
    reduction: str
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    pred = net(x, s)
    if pred.shape[-1:] == (1,):
        pred = pred.squeeze(-1)
    tgt = target
    if tgt.shape[-1:] == (1,):
        tgt = tgt.squeeze(-1)
    if pred.shape != tgt.shape:
        msg = (
            f"net output (squeezed) shape {tuple(pred.shape)} does not match "
            f"target (squeezed) shape {tuple(tgt.shape)}"
        )
        raise ValueError(msg)
    per_sample = (pred - tgt).pow(2)
    return reduce_loss(per_sample, reduction)
