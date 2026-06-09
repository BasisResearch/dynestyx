# WSP

::: dynestyx.models.wsp.WSP
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Weighted Sums Parameterization"
    `WSP` wraps a continuous-time SDE so its solution provably stays inside an
    axis-aligned [`Box`](box.md). Near the box faces the diffusion vanishes and the
    drift points inward toward the box center; in the interior the original dynamics
    are recovered. It implements the box specialization of the Weighted Sums
    Parameterization of Lu, Liu, Nock & Yacoby, *Neural Stochastic Differential
    Equations on Compact State Spaces* (ProbML 2026).

    The result is an ordinary
    [`ContinuousTimeStateEvolution`](../core/continuous_time_state_evolution.md), so it
    composes with all simulators and inference backends. Because the diffusion is
    degenerate at the boundary, prefer ensemble/particle filters (EnKF/DPF) there; see
    [FilterConfigs](../../../inference/filter_configs.md).

## Example

??? example "Constrain an OU process to $[0, 1]$"
    ```python
    import jax.numpy as jnp
    from dynestyx import (
        WSP,
        Box,
        ContinuousTimeStateEvolution,
        ScalarDiffusion,
    )

    # Inner OU SDE: dX_t = theta (mu - X_t) dt + sigma dW_t
    inner = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: 1.0 * (0.5 - x),
        diffusion=ScalarDiffusion(0.45, bm_dim=1),
    )

    # Constrain the solution to the unit interval [0, 1].
    constrained = WSP(
        inner,
        Box(jnp.array([0.0]), jnp.array([1.0])),
        alpha=6.0,
        beta=25.0,
        gamma=1.5,
        epsilon=0.05,
    )
    ```

See the [SDEs on compact spaces with WSP](../../../../deep_dives/wsp_box_constrained_sde.ipynb)
deep dive for a worked example with simulation, filtering, and SVI.
