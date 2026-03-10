from __future__ import annotations

import pytest
import torch

from nami.paths.base import ProbabilityPath


class TestProbabilityPathInterface:
    def setup_method(self):
        self.path = ProbabilityPath()
        self.x = torch.randn(2, 3)
        self.t = torch.rand(2)

    def test_sample_xt_raises(self):
        with pytest.raises(NotImplementedError):
            self.path.sample_xt(self.x, self.x, self.t)

    def test_target_ut_raises(self):
        with pytest.raises(NotImplementedError):
            self.path.target_ut(self.x, self.x, self.t)

    def test_score_target_raises(self):
        with pytest.raises(NotImplementedError):
            self.path.score_target(self.x, self.x, self.t, xt=self.x)
