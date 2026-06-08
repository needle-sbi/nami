r"""Scalar-potential field :math:`\phi(x, \theta)` for parameter-flow.

The curl-free gauge realisation is that the transport velocity is the
spatial gradient :math:`v = \nabla_x \phi`.  The scalar-potential field follows the
:math:`v = \nabla\phi` ansatz, which is the Otto-horizontal representative in the
:math:`\mathrm{Diff}(M)/\mathrm{Diff}_\mu(M)` bundle. It is what makes
parameter-flow transport a Wasserstein geodesic.

Follows the ``(x, t, c)`` calling convention of
:class:`~nami.fields.action.ActionHead` with ``t`` accepted-and-ignored
(:math:`\phi` depends on position and parameter only; the path
parameter ``s`` enters through :math:`\theta(s)`) and ``c`` carrying
:math:`\theta`.

The training loss's Laplacian :math:`\Delta_x\phi` is the same
operator as the runtime ``log_prob`` divergence
(:math:`\nabla\!\cdot v = \nabla\!\cdot\nabla\phi = \Delta\phi`), with
the identical cost structure.  :meth:`velocity_field`
exposes the gradient as a field-shaped object so both consume one
:mod:`nami.divergence` estimator instead of bespoke double-autograd
helpers.

TODO: add any useful refs.
"""

from __future__ import annotations

import torch
from torch import nn

from nami.components import MLPBackbone
from nami.core.specs import TensorSpec, flatten_event, validate_shapes
from nami.fields._common import normalise_event_shape, validate_context


class ScalarPotentialField(nn.Module):
    r"""Scalar potential :math:`\phi(x, \theta) \to \mathbb{R}` per sample.

    Structurally a sibling of :class:`~nami.fields.action.ActionHead`
    (scalar-out MLP head, velocity recovered by autograd), but without a
    time embedding: parameter-flow's "time" is the path parameter ``s``,
    which reaches :math:`\phi` only through :math:`\theta(s)`.

    TODO: any embeddings needed for theta?

    Parameters
    ----------
    dim : int or tuple[int, ...]
        Data dimensionality (event shape).
    theta_dim : int
        Parameter dimensionality (last axis of ``theta``).  Must be
        positive — :math:`\phi(x, \theta)` is meaningless without
        :math:`\theta`.
    hidden : int
        Hidden layer width.
    layers : int
        Number of hidden layers.
    activation : str
        Activation function.
    dropout : float
        Dropout probability.
    layer_norm : bool
        Whether to apply layer normalisation.
    """

    def __init__(
        self,
        dim: int | tuple[int, ...],
        *,
        theta_dim: int,
        hidden: int = 128,
        layers: int = 4,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if theta_dim <= 0:
            msg = f"theta_dim must be positive, got {theta_dim}"
            raise ValueError(msg)

        self.spec = TensorSpec(normalise_event_shape(dim))
        self.theta_dim = int(theta_dim)
        self.backbone = MLPBackbone(
            self.flat_dim + self.theta_dim,
            1,  # scalar output
            hidden=hidden,
            layers=layers,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.spec.event_shape

    @property
    def event_ndim(self) -> int:
        return self.spec.event_ndim

    @property
    def flat_dim(self) -> int:
        return self.spec.numel

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Evaluate the scalar potential :math:`\phi(x, \theta)`.

        ``t`` is accepted for ``(x, t, c)`` convention compatibility and
        ignored; ``c`` carries :math:`\theta`.

        Returns
        -------
        Tensor, shape ``(*lead,)``
            Scalar potential per sample.  Its gradient w.r.t. ``x`` is
            the parameter-flow transport velocity.
        """
        del t
        validate_shapes(x, self.spec)
        x_flat = flatten_event(x, self.event_ndim)
        lead_shape = tuple(x_flat.shape[:-1])
        validate_context(c, self.theta_dim, lead_shape)
        assert c is not None  # narrowed by validate_context (theta_dim > 0)
        inputs = torch.cat([x_flat, c], dim=-1)
        return self.backbone(inputs).squeeze(-1)

    def velocity(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        r"""Recover the velocity :math:`\nabla_x \phi(x, \theta)` by autograd.

        :math:`\phi` is a scalar per sample; summing across the batch
        lets a single ``autograd.grad`` call return the per-sample
        gradient because :math:`\phi_i` does not depend on ``x_j`` for
        ``i != j``.  Always runs under ``torch.enable_grad()`` so it
        works inside ``torch.no_grad()`` sampling loops.

        Parameters
        ----------
        x : torch.Tensor
            Evaluation points, shape ``(*lead, *event_shape)``.
        theta : torch.Tensor
            Parameters, shape ``(*lead, theta_dim)``.
        create_graph : bool
            Build a graph through the gradient so second-order
            objectives (the parameter-flow loss, ``loss.backward()``)
            reach :math:`\phi`'s parameters.  Set ``False`` for
            eval-only transport to save memory.
        """
        with torch.enable_grad():
            if create_graph:
                xx = x if x.requires_grad else x.clone().requires_grad_(True)
            else:
                xx = x.detach().requires_grad_(True)
            phi = self(xx, None, theta)
            (grad_phi,) = torch.autograd.grad(
                outputs=phi.sum(),
                inputs=xx,
                create_graph=create_graph,
            )
        return grad_phi

    def velocity_field(self) -> _GradientField:
        r"""Expose :math:`\nabla_x \phi` as a field-shaped object.

        The returned adapter follows the ``(x, t, c)`` convention and
        ``event_ndim``/``spec`` protocol, so :mod:`nami.divergence`
        estimators applied to it compute
        :math:`\nabla\!\cdot\nabla\phi = \Delta_x\phi`. The single
        shared operator is used by both the parameter-flow loss (its
        Laplacian term) and the runtime log-density divergence.

        The adapter always builds the inner autograd graph
        (``create_graph=True`` on the gradient): its sole purpose is to
        be differentiated again by a divergence estimator, which is
        impossible against a detached gradient.  Whether the resulting
        Laplacian itself carries a graph (for training) is the
        *estimator's* ``create_graph`` flag.
        """
        return _GradientField(self)


class _GradientField:
    """Adapter presenting ``\\nabla_x phi`` as a ``(x, t, c)`` field.

    Private adapter class, not to be exposed.
    """

    def __init__(self, potential: ScalarPotentialField):
        self._potential = potential

    @property
    def spec(self) -> TensorSpec:
        return self._potential.spec

    @property
    def event_ndim(self) -> int:
        return self._potential.event_ndim

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._potential.event_shape

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del t
        if c is None:
            msg = "gradient-field evaluation requires theta as the context c"
            raise ValueError(msg)
        return self._potential.velocity(x, c, create_graph=True)
