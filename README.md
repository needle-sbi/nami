  <p align="center">
    <img src="/docs/assets/nami_logo.svg" width="640" alt="Nami logo">
  </p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![codecov](https://codecov.io/gh/LeviSamuelEvans/nami/branch/main/graph/badge.svg)](https://codecov.io/gh/LeviSamuelEvans/nami)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://github.com/LeviSamuelEvans/nami/actions/workflows/docs.yaml/badge.svg)](https://levisamuelevans.github.io/nami/) 

![Development](https://img.shields.io/badge/status-development-orange?style=for-the-badge)


Nami is a small PyTorch **research toolkit** for experimenting with flow-style generative models under a shared API, with a particular interest in SBI applications. It does not aim to act as a production framework and should be paired with an approrpirate orchestration layer. The point is to collect related models behind a common interface, keep dependencies light, and make it easy to prototype objectives, paths, solvers, and applications, particularly SBI.

Much of the design is inspired by the fantastic [Zuko](https://github.com/probabilists/zuko/tree/master) library, especially its lazy binding pattern for conditional models. In `nami`, lazy base distributions use `LazyDistribution`, while flow-matching, diffusion, and generator-matching wrappers use `LazyProcess` to bind context into a runnable process. Fixed bases such as `StandardNormal` do not need to be lazy, but conditional or learned bases can still opt into `LazyDistribution`.

See the [documentation](https://levisamuelevans.github.io/nami/) to get started, with examples, tutorials and the full reference API. You'll also find different experiments exploring a range of ideas and concepts (*warning: some are very domain specific as the core developer team are high-energy physicists*).

### The purpose of nami

There are several very good flow matching libraries, e.g. [torchcfm](https://github.com/atong01/conditional-flow-matching), [rectified-flow](https://github.com/lqiang67/rectified-floww), Meta's [flow_matching](https://github.com/facebookresearch/flow_matching) each with many strengths and cool features. Nami is simply trying to bring several closely related transport-map workflows into one small PyTorch toolkit, with a focus on SBI (Simulation-Based Inference) applications and research. See the seminal [Cranmer, Brehmer, Louppe 2020](https://arxiv.org/abs/1911.01429) for more information on SBI if you're interested!

Today, that includes flow matching, stochastic-interpolant training, operator-centric generator matching, and diffusion sampling utilities. There has also recently been growing interest in discrete versions of these algorithms towards topics such as language modelling, which may be included at a later date, so stay tuned.

> **Note:** Diffusion support is present but still early. nami already provides schedule and sampler wrappers for `eps`, `score`, and `x0` parameterised diffusion models, but the training objective is still external and this part of the API will be included soon. nami began as a collection of projects that I deciced to collect into a toolkit.

## Installation

### Prerequisites

- Python >= 3.10
- [pixi](https://pixi.sh) package manager

### Setup

Clone the repository and run the setup task, which installs nami in editable mode:

```bash
git clone https://github.com/LeviSamuelEvans/nami
cd nami
```

```bash
pixi run setup
```

This creates a conda environment via pixi and installs the package with `pip install -e .`.

### Without pixi

If you prefer not to use pixi, you can install directly with pip (requires PyTorch >= 2.0):

```bash
pip install -e .
```

### Toy Datasets

The additional `nami_toys` package (toy dataset generators for benchmarking) is included in the same repository and installed automatically alongside `nami`:

```python
from nami_toys import GaussianMixture, GaussianShell, TwoMoons, Checkerboard, GaussianRing, TwoSpirals
```

### Development tasks

pixi provides several convenience tasks:

| Command | Description |
|---------|-------------|
| `pixi run test` | Run tests with pytest |
| `pixi run cov` | Run tests with coverage report |
| `pixi run lint` | Lint with ruff |
| `pixi run fmt` | Format with ruff |
| `pixi run typecheck` | Type-check with mypy |
| `pixi run docs` | Build Sphinx documentation |
| `pixi run kernel` | Install a Jupyter kernel named `nami` |
