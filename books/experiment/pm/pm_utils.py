# for 1d and gallery
import math
import torch

def p0(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def p_theta(x, theta):
    return p0(x) * (1 + theta * torch.tanh(x))


def joint_score(x, theta):
    return torch.tanh(x) / (1 + theta * torch.tanh(x))


def spatial_score(x, theta):
    sech2 = 1.0 / torch.cosh(x) ** 2
    return -x + theta * sech2 / (1 + theta * torch.tanh(x))


def sample_p_theta(theta, generator, lo=-8.0, hi=8.0, n=4001):
    grid = torch.linspace(lo, hi, n, dtype=torch.float64)
    pdf = p_theta(grid, theta.double().unsqueeze(-1))
    cdf = torch.cat(
        [
            torch.zeros(pdf.shape[0], 1, dtype=pdf.dtype),
            torch.cumulative_trapezoid(pdf, grid, dim=-1),
        ],
        dim=-1,
        )
    cdf = cdf / cdf[:, -1:]  # normalise
    u = torch.rand(theta.shape[0], 1, dtype=cdf.dtype, generator=generator)
    idx = torch.searchsorted(cdf, u).clamp(1, grid.numel() - 1)
    c0, c1 = cdf.gather(-1, idx - 1), cdf.gather(-1, idx)
    frac = (u - c0) / (c1 - c0).clamp_min(1e-12)
    return (grid[idx - 1] + frac * (grid[idx] - grid[idx - 1])).float()

def v_true(grid, theta):
    """Continuity-equation velocity: v = -(1/p_theta) * int_-inf^x p0 tanh."""
    integrand = p0(grid) * torch.tanh(grid)
    integral = torch.cat(
        [torch.zeros(1), torch.cumulative_trapezoid(integrand, grid)]
    )
    return -integral / p_theta(grid, torch.as_tensor(theta))