<p align="center"><img src="/docs/assets/nami_logo.svg" width="600" alt="Nami logo"></p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![codecov](https://codecov.io/gh/needle-sbi/nami/branch/main/graph/badge.svg)](https://codecov.io/gh/needle-sbi/nami)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://github.com/needle-sbi/nami/actions/workflows/docs.yaml/badge.svg)](https://needle-sbi.github.io/nami/)

![Development](https://img.shields.io/badge/status-development-orange?style=for-the-badge)


Nami is a library for flow-style generative models, with a focus on composable transport-map workflows and SBI applications.

See the [documentation](https://needle-sbi.github.io/nami/) for the full guide, examples, tutorials (UNDERGOING RE-WRITE), and API reference.

## Development

For local development, either install with [pixi](https://pixi.sh):

```bash
git clone https://github.com/needle-sbi/nami
cd nami
pixi run setup # pixi run -e gpu setup
```

or, install directly with `pip` (inside a venv):

```bash
pip install -e . --group dev
```

Note that `--group` requires pip >= 25.1 (run `pip install --upgrade pip` first if needed). On older pip, install the package and dev tools separately with `pip install -e .` followed by `pip install pytest pytest-cov ruff ty hatch`.

For Pixi, common development commands are `pixi run test`, `pixi run lint`, `pixi run fmt`, `pixi run typecheck`, and `pixi run docs`.

### GPU environments

For nvidia GPU workflows, two environments pull `pytorch-gpu` from conda-forge, pinned to `linux-64`, differing only in CUDA major:

- `gpu` (`cuda = "12"`) is the default. cu12 builds also run on cu13+ drivers, so this one env works across mixed-driver batch nodes.
- `gpu-cu13` (`cuda = "13"`) unlocks cu13-only kernels. Requires a CUDA >= 13 driver to solve and run. This option is only for cases where the latest nvidia architectures are being used.

```bash
pixi run -e gpu setup        # cu12 portable
pixi run -e gpu-cu13 setup   # cu13 newest archs
```

Pick `gpu-cu13` only where the partition actually has CUDA >= 13 drivers otherwise `gpu` is the safe choice. See [PyTorch v2.12.0](https://github.com/pytorch/pytorch/releases/tag/v2.12.0) (cu128 dropped, cu130 now the default, cu126 retained) and the [2.12 blog](https://pytorch.org/blog/pytorch-2-12-release-blog/).

Each environment declares a `system-requirements` CUDA floor, so resolving or locking it requires a matching driver. On a system without an nvidia GPU, set `CONDA_OVERRIDE_CUDA` to solve/lock for `linux-64`:

```bash
CONDA_OVERRIDE_CUDA=12 pixi install -e gpu        # also for `pixi lock`, `pixi info`
CONDA_OVERRIDE_CUDA=13 pixi install -e gpu-cu13
```

> [!NOTE]
> Intel GPUs (XPU). The conda-forge `pytorch-gpu` build (and hence the `gpu` Pixi environment) is CUDA-only, so it does not cover Intel GPUs. XPU support is native in PyTorch >= 2.5 (via `torch.xpu`); note this is newer than the project's `torch >= 2.0` floor, so ensure you install a 2.5+ build. Install from PyTorch's XPU wheel index instead of the `gpu` environment:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/xpu
> pip install -e . --no-deps --group dev   # reuse if already installed
> ```
>
> Verify with `torch.xpu.is_available()`.

### Jupyter Notebooks

When using notebooks, register a kernel like:

```bash
pixi run kernel                  # cpu/mps registers the "nami (CPU)" kernel
pixi run -e gpu kernel-gpu       # linux + nvidia, registers the "nami (GPU, cu12)" kernel
pixi run -e gpu-cu13 kernel-gpu  # linux + nvidia (cuda >= 13), registers the "nami (GPU, cu13)" kernel
```

Open a notebook and select the `nami (CPU)` or `nami (GPU)` kernel via either:

- `VS Code`: open a `.ipynb`, and pick the kernel from the kernel selector in the top-right. (requires the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter))
- `JupyterLab`: bundled in the default Pixi environment, so launch it with `pixi run jupyter lab` and select the kernel from the launcher or "Kernel -> Change Kernel".


The two kernels are registered under different names to prevent clobbering. You can confirm the active device, and that `nami` is importable, with:

```python
import torch
print(torch.cuda.is_available())                 # true on cuda
print(torch.backends.mps.is_available())         # true on apple silicon

import nami
print(nami.__version__)                           # nami imports torch, so this also exercises the torch install
```

TODO: short comments on marimo

---

Supported by HelmholtzAI and DESY.

[The NEEDLE project](https://needle-sbi.github.io/)

<img src="/docs/_static/institutes/Helmholtz-Logo-Blue-RGB.png" alt="HelmholtzAI" width="160"/> &nbsp;
