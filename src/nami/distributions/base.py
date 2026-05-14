"""Distribution-handling helpers shared across processes."""

from __future__ import annotations


from torch.distributions import Distribution


def expand_distribution(
    dist: Distribution, batch_shape: tuple[int, ...]
) -> Distribution:
    """Expand ``dist`` to ``batch_shape`` (no-op when already matching)."""
    if dist.batch_shape == batch_shape:
        return dist
    if not hasattr(dist, "expand"):
        msg = "distribution does not support expand"
        raise ValueError(msg)
    return dist.expand(batch_shape)


def has_rsample(dist: Distribution) -> bool:
    """Return ``True`` if ``dist`` supports the reparameterised ``rsample``."""
    return bool(getattr(dist, "has_rsample", False))
