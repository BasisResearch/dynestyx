# Diffusion

`Diffusion` objects define the stochastic term in a continuous-time state evolution

\[
dx_t = f(x_t, u_t, t)\,dt + L(x_t, u_t, t)\,dW_t,
\]

where:

- \(x_t \in \mathbb{R}^{d_x}\) is the latent state,
- \(u_t\) is an optional control,
- \(W_t \in \mathbb{R}^{d_w}\) is Brownian motion,
- `bm_dim = d_w`,
- and \(L(x_t, u_t, t)\) is the diffusion coefficient.

Dynestyx exposes three structured diffusion classes:

- `FullDiffusion`: general matrix-valued diffusion \(L \in \mathbb{R}^{d_x \times d_w}\)
- `DiagonalDiffusion`: vector-valued diffusion \(v \in \mathbb{R}^{d_x}\)
- `ScalarDiffusion`: scalar-valued diffusion \(\sigma \in \mathbb{R}\)

In practice, many models use a constant diffusion coefficient. In those cases,
pass the matrix/vector/scalar value directly. Reserve callable diffusion
coefficients for cases where the coefficient genuinely depends on state,
control, or time.

## Mathematical Interpretation

### `FullDiffusion`

`FullDiffusion` represents an arbitrary matrix-valued diffusion coefficient:

\[
L(x_t, u_t, t) \in \mathbb{R}^{d_x \times d_w}.
\]

The SDE is

\[
dx_t = f(x_t, u_t, t)\,dt + L(x_t, u_t, t)\,dW_t.
\]

This is the most general case. The process noise covariance is

\[
L(x_t, u_t, t)\,L(x_t, u_t, t)^\top,
\]

which may be dense.

### `DiagonalDiffusion`

`DiagonalDiffusion` represents a vector-valued coefficient

\[
v(x_t, u_t, t) \in \mathbb{R}^{d_x}.
\]

Its interpretation depends on `bm_dim`.

If `bm_dim = d_x`, the vector is interpreted as a diagonal matrix:

\[
L(x_t, u_t, t) = \mathrm{diag}(v(x_t, u_t, t)),
\]

so each state coordinate receives its own independent Brownian driver and

\[
L L^\top = \mathrm{diag}(v_1^2, \ldots, v_{d_x}^2).
\]

If `bm_dim = 1`, the same vector is interpreted as a column loading vector:

\[
L(x_t, u_t, t) = v(x_t, u_t, t) \in \mathbb{R}^{d_x \times 1},
\]

so all state coordinates share a single Brownian path. In that case,

\[
L L^\top = v v^\top,
\]

which is rank 1.

### `ScalarDiffusion`

`ScalarDiffusion` represents a scalar-valued coefficient

\[
\sigma(x_t, u_t, t) \in \mathbb{R}.
\]

Its interpretation also depends on `bm_dim`.

If `bm_dim = d_x`, it is interpreted as isotropic independent noise:

\[
L(x_t, u_t, t) = \sigma(x_t, u_t, t)\,I_{d_x},
\]

so the process noise covariance is

\[
L L^\top = \sigma^2 I_{d_x}.
\]

If `bm_dim = 1`, it is interpreted as a shared scalar driver applied equally to
every state coordinate:

\[
L(x_t, u_t, t) = \sigma(x_t, u_t, t)\,\mathbf{1}_{d_x},
\]

viewed as a column vector in \(\mathbb{R}^{d_x \times 1}\). The resulting
covariance is

\[
L L^\top = \sigma^2 \mathbf{1}_{d_x}\mathbf{1}_{d_x}^\top,
\]

which is again rank 1.

## Typical Usage

```python
import jax.numpy as jnp
from dynestyx import ContinuousTimeStateEvolution, FullDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=FullDiffusion(jnp.eye(2)),
)
```

```python
from dynestyx import DiagonalDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=DiagonalDiffusion(jnp.array([0.1, 0.2]), bm_dim=2),
)
```

```python
from dynestyx import ScalarDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=ScalarDiffusion(0.1, bm_dim=2),
)
```

## API

::: dynestyx.models.diffusions.Diffusion
    options:
      show_root_heading: false

### `FullDiffusion`

::: dynestyx.models.diffusions.FullDiffusion
    options:
      show_root_heading: false

### `DiagonalDiffusion`

::: dynestyx.models.diffusions.DiagonalDiffusion
    options:
      show_root_heading: false

### `ScalarDiffusion`

::: dynestyx.models.diffusions.ScalarDiffusion
    options:
      show_root_heading: false
