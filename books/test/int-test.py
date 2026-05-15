"""Demonstrator notebook for the unified Interpolant + Parameterization vocabulary.

Run with ``marimo edit books/test/int-test.py``.

Each cell is a small self-contained example. Top section walks the
happy paths (one cell per Interpolant x Parameterization x Process
triple); bottom section exercises the guardrails — calls that should
fail with a useful message rather than a cryptic AttributeError.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    import marimo as mo
    import nami

    torch.manual_seed(0)
    return mo, nami, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    # Interpolant + Parameterization demonstrators

    Two axes:

    1. **Vocabulary axis** — combinations of `Interpolant`, `Parameterization`,
       and `Process` that should work end-to-end (loss runs, samples come out).
    2. **Guardrail axis** — combinations that *should* fail at construction
       or at `forward()` with a message pointing at the right alternative.

    Each cell is self-contained. Re-running one cell does not depend on
    the others having run.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("## Happy paths")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 1. Flow matching — `LinearInterpolant` + `velocity_prediction`

    The default flow-matching recipe: deterministic linear path between
    `x_data` and `x_noise`, velocity target, no schedule needed.
    """
    )
    return


@app.cell
def _(nami, torch):
    field_fm = nami.VelocityField(dim=4, hidden=32, layers=1)
    loss_fm = nami.regression_loss(
        field_fm,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.LinearInterpolant(),
        parameterization=nami.velocity_prediction(),
    )
    proc_fm = nami.FlowMatching(field_fm, nami.StandardNormal(4), nami.RK4(steps=4))()
    samples_fm = proc_fm.sample((3,))
    print(f"FlowMatching: loss={loss_fm.item():.3f}  samples={tuple(samples_fm.shape)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 2. Diffusion — `GaussianInterpolant(VPSchedule)` + `epsilon_prediction`

    The Gaussian interpolant *needs* a schedule (it relies on `\alpha(t)`,
    `\sigma(t)` to form `x_t`). Same field architecture; the loss now
    targets `\epsilon` with the conventional weighting.
    """
    )
    return


@app.cell
def _(nami, torch):
    schedule = nami.VPSchedule()
    field_eps = nami.VelocityField(dim=4, hidden=32, layers=1)
    loss_eps = nami.regression_loss(
        field_eps,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.GaussianInterpolant(schedule),
        parameterization=nami.epsilon_prediction(schedule),
    )
    proc_eps = nami.Diffusion(
        field_eps,
        schedule,
        nami.EulerMaruyama(steps=8),
        parameterization=nami.epsilon_prediction(schedule),
        base=nami.StandardNormal(4),
    )()
    samples_eps = proc_eps.sample((3,))
    print(f"Diffusion(eps): loss={loss_eps.item():.3f}  samples={tuple(samples_eps.shape)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 3. Same diffusion, swap target — `score_prediction` / `x0_prediction`

    Three target choices, three different loss values, but the **sampling
    trajectory is identical** under matched `\epsilon` emission (the
    structural guarantee from the Diffusion Process pattern-matching on
    `parameterization.target`).
    """
    )
    return


@app.cell
def _(nami, torch):
    sched = nami.VPSchedule()
    field_x0 = nami.VelocityField(dim=4, hidden=32, layers=1)
    field_score = nami.VelocityField(dim=4, hidden=32, layers=1)

    loss_x0 = nami.regression_loss(
        field_x0,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.GaussianInterpolant(sched),
        parameterization=nami.x0_prediction(sched),
    )
    loss_score = nami.regression_loss(
        field_score,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.GaussianInterpolant(sched),
        parameterization=nami.score_prediction(sched),
    )
    print(f"x0_prediction:    loss={loss_x0.item():.3f}")
    print(f"score_prediction: loss={loss_score.item():.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 4. Generator matching — `BrownianBridgeInterpolant` + `generator_prediction`

    The operator now lives inside the `Parameterization` (carried by the
    `GeneratorParams` target). The Process consults
    `parameterization.output_transform` for projection, so a future
    constrained projection drops in without touching the Process.
    """
    )
    return


@app.cell
def _(nami, torch):
    op = nami.ItoGeneratorOperator(event_shape=4, diffusion="diagonal")
    field_gm = nami.GeneratorField(dim=4, operator=op, hidden=32, layers=1)
    param_gm = nami.generator_prediction(op)

    loss_gm = nami.regression_loss(
        field_gm,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.BrownianBridgeInterpolant(),
        parameterization=param_gm,
        eps_t=1e-3,
    )
    proc_gm = nami.GeneratorMatching(
        field_gm,
        nami.EulerMaruyama(steps=8),
        parameterization=param_gm,
        base=nami.StandardNormal(4),
    )()
    samples_gm = proc_gm.sample((3,))
    print(f"GeneratorMatching: loss={loss_gm.item():.3f}  samples={tuple(samples_gm.shape)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5. Stochastic interpolants — `StochasticLinearInterpolant`

    Adds a `\gamma(t) z` noise term to the linear path. Same
    `velocity_prediction` target; the `z` noise is auto-drawn unless
    passed explicitly.
    """
    )
    return


@app.cell
def _(nami, torch):
    field_sfm = nami.VelocityField(dim=4, hidden=32, layers=1)
    loss_sfm = nami.regression_loss(
        field_sfm,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.StochasticLinearInterpolant(),
        parameterization=nami.velocity_prediction(),
        eps_t=1e-3,
    )
    print(f"StochasticLinear: loss={loss_sfm.item():.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 6. Consistency loss — trajectory-pair MSE

    A separate loss family (not `regression_loss`). Samples two
    trajectory points and MSEs the consistency function between them.
    `target_time` must be exactly `0.0` (forward) or `1.0` (reverse) —
    intermediate values are rejected.
    """
    )
    return


@app.cell
def _(nami, torch):
    field_cfm = nami.VelocityField(dim=4, hidden=32, layers=1)
    target_field = nami.VelocityField(dim=4, hidden=32, layers=1)
    target_field.load_state_dict(field_cfm.state_dict())

    loss_cfm = nami.consistency_loss(
        field_cfm,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.LinearInterpolant(),
        parameterization=nami.velocity_prediction(),
        target_field=target_field,
        delta=0.05,
        target_time=0.0,
    )
    print(f"consistency_loss (forward): loss={loss_cfm.item():.3f}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 7. Conditional flow matching

    Identical to (1) but the field accepts a `condition_dim` and the
    loss / process receive a context `c`.
    """
    )
    return


@app.cell
def _(nami, torch):
    c = torch.randn(8, 2)
    cfield = nami.VelocityField(dim=4, condition_dim=2, hidden=32, layers=1)
    loss_c = nami.regression_loss(
        cfield,
        torch.randn(8, 4),
        torch.randn(8, 4),
        interpolant=nami.LinearInterpolant(),
        parameterization=nami.velocity_prediction(),
        c=c,
    )
    proc_c = nami.FlowMatching(cfield, nami.StandardNormal(4), nami.RK4(steps=4))(c)
    samples_c = proc_c.sample((1,))
    print(f"Conditional FM: loss={loss_c.item():.3f}  samples={tuple(samples_c.shape)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Guardrails — these *should* fail

    Every cell below is wrapped in a `try/except` so the notebook
    still runs top-to-bottom. The point is to inspect the error
    *message* — it should name the right alternative, not just the
    abstract reason.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G1. `GaussianInterpolant(4)` — passing a dim instead of a schedule

    A common first-use trap. Now caught at construction with a
    `TypeError` pointing at `nami.VPSchedule()` (and at
    `LinearInterpolant` for the no-schedule case).
    """
    )
    return


@app.cell
def _(nami):
    try:
        nami.GaussianInterpolant(4)
    except TypeError as err:
        print(f"TypeError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G2. `GaussianInterpolant + Velocity` — unsupported target

    Velocity needs `\alpha'(t)`, `\sigma'(t)` which `NoiseSchedule`
    does not expose. The error names the right alternatives.
    """
    )
    return


@app.cell
def _(nami, torch):
    from nami.interpolants.protocol import InterpolantState

    interp = nami.GaussianInterpolant(nami.VPSchedule())
    state = interp.sample(torch.randn(4, 4), torch.randn(4, 4), torch.full((4,), 0.3))
    try:
        interp.target(nami.Velocity(), state)
    except NotImplementedError as err:
        print(f"NotImplementedError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G3. Legacy string flag on `Diffusion`

    Pre-refactor users passed `parameterization="eps"`. The new API
    rejects the string with a migration message.
    """
    )
    return


@app.cell
def _(nami):
    field_g3 = nami.VelocityField(dim=4, hidden=32, layers=1)
    try:
        nami.Diffusion(
            field_g3,
            nami.VPSchedule(),
            nami.EulerMaruyama(steps=4),
            parameterization="eps",  # type: ignore[arg-type]
        )()
    except TypeError as err:
        print(f"TypeError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G4. `LinearInterpolant + Score` — wrong path family for the target

    Score needs a schedule (`-\epsilon / \sigma(t)`). LinearInterpolant
    is deterministic and schedule-free.
    """
    )
    return


@app.cell
def _(nami, torch):
    interp_lin = nami.LinearInterpolant()
    state_lin = interp_lin.sample(torch.randn(4, 4), torch.randn(4, 4), torch.full((4,), 0.3))
    try:
        interp_lin.target(nami.Score(), state_lin)
    except NotImplementedError as err:
        print(f"NotImplementedError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G5. `consistency_loss` with intermediate `target_time`

    Only `0.0` (forward consistency) and `1.0` (reverse) are valid
    anchors. Anything in between would silently misuse the fixed
    `< 0.5` anchor-split heuristic.
    """
    )
    return


@app.cell
def _(nami, torch):
    field_g5 = nami.VelocityField(dim=4, hidden=32, layers=1)
    try:
        nami.consistency_loss(
            field_g5,
            torch.randn(8, 4),
            torch.randn(8, 4),
            interpolant=nami.LinearInterpolant(),
            parameterization=nami.velocity_prediction(),
            target_field=field_g5,
            delta=0.05,
            target_time=0.5,
        )
    except ValueError as err:
        print(f"ValueError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### G6. Negative `delta` in `consistency_loss`

    Would push `t + delta` below 0; the bridge interpolant's
    `sqrt(t(1-t))` of a negative argument would `nan`-out silently.
    """
    )
    return


@app.cell
def _(nami, torch):
    field_g6 = nami.VelocityField(dim=4, hidden=32, layers=1)
    try:
        nami.consistency_loss(
            field_g6,
            torch.randn(8, 4),
            torch.randn(8, 4),
            interpolant=nami.LinearInterpolant(),
            parameterization=nami.velocity_prediction(),
            target_field=field_g6,
            delta=-0.05,
            target_time=0.0,
        )
    except ValueError as err:
        print(f"ValueError: {err}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Equivalence check — three diffusion targets, one trajectory

    The structural claim behind the Diffusion refactor: changing the
    target only changes how the network is parameterised — given fields
    that emit the same underlying `\epsilon`, the sampled trajectory
    must be identical (up to float precision).

    Below we build three constant fields that all emit the same
    `\epsilon = 0` in their respective parameter spaces and check the
    samples coincide.
    """
    )
    return


@app.cell
def _(nami, torch):
    class ConstField(torch.nn.Module):
        event_ndim = 1

        def __init__(self, value: torch.Tensor):
            super().__init__()
            self.register_buffer("value", value)

        def forward(self, x, t, c=None):
            return self.value.expand_as(x)

    sched_eq = nami.VPSchedule()
    eps_field = ConstField(torch.zeros(4))
    x0_field = ConstField(torch.zeros(4))
    score_field = ConstField(torch.zeros(4))

    base = nami.StandardNormal(4)
    solver = nami.EulerMaruyama(steps=8)
    torch.manual_seed(0)
    s_eps = nami.Diffusion(eps_field, sched_eq, solver, parameterization=nami.epsilon_prediction(sched_eq), base=base)().sample((4,))
    torch.manual_seed(0)
    s_x0 = nami.Diffusion(x0_field, sched_eq, solver, parameterization=nami.x0_prediction(sched_eq), base=base)().sample((4,))
    torch.manual_seed(0)
    s_score = nami.Diffusion(score_field, sched_eq, solver, parameterization=nami.score_prediction(sched_eq), base=base)().sample((4,))

    print(f"max |s_eps - s_x0|    = {(s_eps - s_x0).abs().max().item():.2e}")
    print(f"max |s_eps - s_score| = {(s_eps - s_score).abs().max().item():.2e}")
    return


if __name__ == "__main__":
    app.run()
