import io
import base64

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import numpy as np
from IPython.display import display, HTML


# ---------------------------------------------------------------------------------
# HTML figure helper to convert matplotlib figures to centered HTML for notebook display

def _show_fig(fig, dpi=150):
    """Render matplotlib figure as centered HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    display(HTML(
        f"<div style='text-align:center'>"
        f"<img src='data:image/png;base64,{img_b64}'>"
        f"</div>"
    ))


# ---------------------------------------------------------------------------------
# Flow diagram for basic noise-to-data transformation visualisation

def _flow_diagram():
    """Noise to Data schematic."""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    ax.set_aspect("equal"); ax.axis("off")
    for x, label, c in [(1.5, r"$z \sim \mathcal{N}(0,I)$", "#6baed6"),
                         (8.5, r"$x \sim p_{\mathrm{data}}$", "#fd8d3c")]:
        ax.add_patch(plt.Circle((x, 1.0), 0.7, color=c, alpha=0.35))
        ax.text(x, 1.0, label, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(7.6, 1.0), xytext=(2.4, 1.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))
    ax.text(5.0, 1.3, "learned transform", ha="center", va="bottom",
            fontsize=10, style="italic")
    plt.tight_layout()
    _show_fig(fig)


# ---------------------------------------------------------------------------------
# Interpolation path for linear interpolation visualistion with velocity arrows

def _interpolation_path():
    """Visualise a linear interpolation path with velocity arrows."""
    x_data = torch.tensor([2.0, 1.0])
    x_noise = torch.tensor([-1.0, -1.0])

    times = torch.linspace(0, 1, 51)
    path = torch.stack([(1 - t) * x_noise + t * x_data for t in times])
    velocity = (x_data - x_noise).numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    # path
    ax.plot(path[:, 0], path[:, 1], color="#555555", lw=2, zorder=2)

    # gradient-coloured dots along path
    cmap = plt.cm.coolwarm
    for i, t_val in enumerate(times):
        ax.scatter(
            path[i, 0], path[i, 1],
            c=[cmap(t_val.item())], s=18, zorder=3, edgecolors="none",
        )

    # velocity arrows at a few points
    arrow_idx = np.linspace(0, len(times) - 1, 7, dtype=int)[:-1]
    for i in arrow_idx:
        ax.annotate(
            "", xy=(path[i, 0] + velocity[0] * 0.18,
                    path[i, 1] + velocity[1] * 0.18),
            xytext=(path[i, 0], path[i, 1]),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#7b2d8e"),
            zorder=4,
        )

    # endpoints
    ax.scatter(*x_data, c="#2ca02c", s=220, marker=".", zorder=5,
               edgecolors="white", linewidths=0.6, label=r"$t=1$ (data)")
    ax.scatter(*x_noise, c="#d62728", s=120, marker="o", zorder=5,
               edgecolors="white", linewidths=0.6, label=r"$t=0$ (source / noise)")

    # labels
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_title(
        r"Linear interpolation path  $x_t = (1-t)\,x_{\mathrm{noise}} + t\,x_{\mathrm{data}}$",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    _show_fig(fig)

# ---------------------------------------------------------------------------------
# GIF helper to Convert matplotlib animations to inline GIF for notebook display

def _show_gif(anim, fps=30):
    """Render FuncAnimation as centered inline gif."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "anim.gif"
        anim.save(str(path), writer=PillowWriter(fps=fps), dpi=120)
        plt.close(anim._fig)
        gif_bytes = path.read_bytes()
    img_b64 = base64.b64encode(gif_bytes).decode()
    display(HTML(
        f"<div style='text-align:center'>"
        f"<img src='data:image/gif;base64,{img_b64}'>"
        f"</div>"
    ))

# ---------------------------------------------------------------------------------
# flow field animation helper to animate particles and velocity field from noise to data

def _flow_field_gif(n_particles=200, n_frames=60, seed=42, n_grid=14):
    """Animate particles and velocity field from noise to data.

    Left panel: particles transported from source to target via the linear
    interpolant x_t = (1-t) x_0 + t x_1.
    Right panel: the conditional velocity field u_t = x_0 - x_1 evaluated on
    a grid, showing how the field varies with the interpolant position at
    each time step.

    Parameters
    ----------
    n_particles : int
        Number of particles to animate.
    n_frames : int
        Number of animation frames.
    seed : int
        Random seed for reproducibility.
    n_grid : int
        Number of grid points per axis for the quiver plot.
    """
    rng = np.random.RandomState(seed)

    n_half = n_particles // 2
    x0_a = rng.randn(n_half, 2) * 0.35 + np.array([1.5, 1.5])
    x0_b = rng.randn(n_particles - n_half, 2) * 0.35 + np.array([-1.5, -1.0])
    x0 = np.concatenate([x0_a, x0_b], axis=0)

    x1 = rng.randn(n_particles, 2)

    # velocity is constant for linear interpolation: u = x0 - x1
    velocity = x0 - x1  # (n_particles, 2)

    # time runs from 1 (noise) to 0 (data) = sampling direction
    t_vals = np.linspace(1.0, 0.0, n_frames)

    lim = 4.0
    gx = np.linspace(-lim, lim, n_grid)
    gy = np.linspace(-lim, lim, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)  # (n_grid^2, 2)

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(12, 5))

    for ax in (ax_p, ax_v):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=9)
        ax.set_xlabel(r"$x_1$", fontsize=10)
        ax.set_ylabel(r"$x_2$", fontsize=10)

    ax_p.scatter(x0[:, 0], x0[:, 1], c="#2ca02c", s=10, alpha=0.10,
                 edgecolors="none", label="target")
    ax_p.scatter(x1[:, 0], x1[:, 1], c="#d62728", s=10, alpha=0.10,
                 edgecolors="none", label="source")
    cmap = plt.cm.coolwarm
    scat = ax_p.scatter([], [], s=16, edgecolors="none", zorder=5)
    ax_p.legend(fontsize=8, loc="upper left", framealpha=0.8, markerscale=2)
    title_p = ax_p.set_title("", fontsize=11)

    quiv = ax_v.quiver(
        GX, GY, np.zeros_like(GX), np.zeros_like(GY),
        color="#555555", alpha=0.8, scale=25, width=0.004, zorder=3,
    )
    # faint particle positions on velocity panel too
    scat_v = ax_v.scatter([], [], s=8, alpha=0.25, edgecolors="none", zorder=4)
    title_v = ax_v.set_title("", fontsize=11)

    plt.tight_layout()

    def _update(frame):
        t = t_vals[frame]
        xt = (1 - t) * x0 + t * x1

        scat.set_offsets(xt)
        scat.set_color([cmap(t)] * n_particles)
        title_p.set_text(rf"Particles  $t = {t:.2f}$")

        # velocity field via nearest-neighbour interpolation
        # for each grid point, find nearest particle and use its velocity
        from scipy.spatial import cKDTree
        tree = cKDTree(xt)
        _, idx = tree.query(grid_pts, k=3)
        u_grid = velocity[idx].mean(axis=1)
        quiv.set_UVC(u_grid[:, 0].reshape(GX.shape),
                     u_grid[:, 1].reshape(GY.shape))
        scat_v.set_offsets(xt)
        scat_v.set_color([cmap(t)] * n_particles)
        title_v.set_text(rf"Velocity field  $u_t = x_0 - x_1$")
        return scat, title_p, quiv, scat_v, title_v

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    _show_gif(anim, fps=30)


# ---------------------------------------------------------------------------------
# toy data generator of two interleaving half circles (moons dataset)

def sample_moons(n_samples, noise=0.1):
    """Generate two interleaving half circles (moons dataset)."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # outer moon
    outer_circ_x = torch.cos(torch.linspace(0, np.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, np.pi, n_samples_out))
    
    # inner moon
    inner_circ_x = 1 - torch.cos(torch.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 0.5 - torch.sin(torch.linspace(0, np.pi, n_samples_in))
    
    X = torch.vstack([
        torch.stack([outer_circ_x, outer_circ_y], dim=1),
        torch.stack([inner_circ_x, inner_circ_y], dim=1)
    ])
    
    # add noise and suffle
    X = X + torch.randn_like(X) * noise
    X = X[torch.randperm(len(X))]
    
    return X

# ---------------------------------------------------------------------------------
# Learned flow visualisation helper to animate streamlines of trained velocity field

def _learned_flow_gif(
    field,
    device,
    n_particles=300,
    n_steps=40,
    grid_density=20,
    xlim=(-3.5, 3.5),
    ylim=(-3.0, 3.0),
    fps=20,
    dpi=100,
):
    """Animate streamlines of a learned velocity field with transported particles.

    Integrates particles from t=1 (noise) to t=0 (data) via Euler steps using
    the trained ``field`` network.  At each frame the velocity field is evaluated
    on a regular grid and visualised with ``ax.streamplot``.

    Parameters
    ----------
    field : callable
        Trained velocity network with signature ``field(x, t)`` where *x* has
        shape ``(N, 2)`` and *t* has shape ``(N,)``.
    device : torch.device | str
        Device the model lives on.
    n_particles : int
        Number of particles to transport.
    n_steps : int
        Number of Euler integration steps (= number of GIF frames).
    grid_density : int
        Number of grid points per axis for the streamline plot.
    xlim, ylim : tuple[float, float]
        Axis limits.
    fps : int
        Frames per second for the output GIF.
    dpi : int
        Resolution of each frame.
    """
    from PIL import Image as _PILImage
    x = torch.randn(n_particles, 2, device=device)
    dt = -1.0 / n_steps 
    gx = np.linspace(xlim[0], xlim[1], grid_density)
    gy = np.linspace(ylim[0], ylim[1], grid_density)
    GX, GY = np.meshgrid(gx, gy)
    grid_flat = torch.tensor(
        np.column_stack([GX.ravel(), GY.ravel()]),
        dtype=torch.float32,
        device=device,
    )

    frames: list[_PILImage.Image] = []
    t = 1.0

    plt.ioff()
    for step in range(n_steps + 1):
        t_grid = torch.full((len(grid_flat),), t, dtype=torch.float32, device=device)
        with torch.no_grad():
            v_grid = field(grid_flat, t_grid)
        U = v_grid[:, 0].cpu().numpy().reshape(GX.shape)
        V = v_grid[:, 1].cpu().numpy().reshape(GY.shape)
        speed = np.sqrt(U**2 + V**2)

        # for drawing the frame
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.streamplot(
            GX, GY, U, V,
            color=speed,
            cmap="magma",
            density=1.5,
            linewidth=1.0,
            arrowstyle="->",
            arrowsize=1.4,
        )

        x_np = x.detach().cpu().numpy()
        ax.scatter(
            x_np[:, 0], x_np[:, 1],
            s=12, c="black", alpha=0.85,
            edgecolors="white", linewidths=0.3, zorder=5,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=11)
        ax.set_ylabel(r"$x_2$", fontsize=11)
        ax.set_title(rf"Learned velocity field   $t = {t:.2f}$", fontsize=12)
        ax.grid(True, alpha=0.12)
        ax.tick_params(labelsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(_PILImage.open(buf).copy())

        if step < n_steps:
            t_tensor = torch.full((n_particles,), t, dtype=torch.float32, device=device)
            with torch.no_grad():
                v = field(x, t_tensor)
            x = x + v * dt
            t = t + dt

    plt.ion()

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    gif_buf.seek(0)
    img_b64 = base64.b64encode(gif_buf.read()).decode()
    display(HTML(
        f"<div style='text-align:center'>"
        f"<img src='data:image/gif;base64,{img_b64}'>"
        f"</div>"
    ))