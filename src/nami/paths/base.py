from __future__ import annotations

import torch


class ProbabilityPath:
    """Base class for probability paths that interpolate between target and source.

    Subclasses implement ``sample_xt`` to draw noisy interpolants and
    ``target_ut`` to return the conditional velocity target used for
    flow-matching losses.  Stochastic paths (e.g. Brownian bridges) may
    additionally override ``score_target`` to provide the conditional score.

    Parameters passed as keyword-only arguments (``z``, ``xt``) are optional
    hooks for deterministic sampling and stochastic correction terms
    respectively.  Deterministic paths may safely ignore them.
    """

    def sample_xt(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def target_ut(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        xt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def score_target(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        *,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
