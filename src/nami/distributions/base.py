"""Distribution-handling helpers shared across processes."""

from __future__ import annotations

from torch.distributions import Distribution


def expand_distribution(
    dist: Distribution, batch_shape: tuple[int, ...]
) -> Distribution:
    """Expand a distribution to a batch shape.

    Args:
        dist (Distribution): Distribution to expand.
        batch_shape (tuple[int, ...]): Desired batch shape.

    Returns:
        Distribution: ``dist`` when its batch shape already matches, otherwise
        ``dist.expand(batch_shape)``.
    """
    if dist.batch_shape == batch_shape:
        return dist
    if not hasattr(dist, "expand"):
        msg = "distribution does not support expand"
        raise ValueError(msg)
    return dist.expand(batch_shape)


def has_rsample(dist: Distribution) -> bool:
    """Check whether a distribution supports reparameterized samples.

    Args:
        dist (Distribution): Distribution to inspect.

    Returns:
        bool: ``True`` when ``dist.has_rsample`` is truthy.
    """
    return bool(getattr(dist, "has_rsample", False))
