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

## Choosing a Diffusion Class

- Use `ScalarDiffusion` when one scalar scale is enough. This is the most common
  choice for isotropic diffusion.
- Use `DiagonalDiffusion` when each state coordinate should have its own scale
  but the loading remains axis-aligned.
- Use `FullDiffusion` when you need to specify a genuinely matrix-valued loading.

For `DiagonalDiffusion` and `ScalarDiffusion`, `bm_dim` determines whether the
state coordinates have independent Brownian drivers (`bm_dim = state_dim`) or
share a single Brownian driver (`bm_dim = 1`).

## Construction

Each diffusion class accepts either:

- a constant coefficient, or
- a callable `(x, u, t) -> value`.

### Constructor Arguments

All three constructors take a first positional argument `coefficient`.

- `FullDiffusion(coefficient, bm_dim=None)`
  - `coefficient` may be:
    - a constant array with trailing shape `(state_dim, bm_dim)`, or
    - a callable `(x, u, t) -> array` with trailing shape `(state_dim, bm_dim)`.
  - `bm_dim` is optional and is inferred automatically from the matrix shape when omitted.

- `DiagonalDiffusion(coefficient, bm_dim)`
  - `coefficient` may be:
    - a constant vector with trailing shape `(state_dim,)`, or
    - a callable `(x, u, t) -> array` with trailing shape `(state_dim,)`.
  - `bm_dim` is required and must be either `1` or `state_dim`.

- `ScalarDiffusion(coefficient, bm_dim)`
  - `coefficient` may be:
    - a scalar,
    - a constant array with trailing shape `(1,)`, or
    - a callable `(x, u, t) -> scalar_or_length_1_array`.
  - `bm_dim` is required and must be either `1` or `state_dim`.

Typical constant constructions look like:

```python
import jax.numpy as jnp
from dynestyx import DiagonalDiffusion, FullDiffusion, ScalarDiffusion

FullDiffusion(jnp.eye(2))
DiagonalDiffusion(jnp.array([0.1, 0.2]), bm_dim=2)
ScalarDiffusion(0.1, bm_dim=2)
```

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

This is the most general case.

### `DiagonalDiffusion`

`DiagonalDiffusion` represents a vector-valued coefficient

\[
v(x_t, u_t, t) \in \mathbb{R}^{d_x}.
\]

Its interpretation depends on `bm_dim`.

If `bm_dim = d_x`, the vector is interpreted as a diagonal matrix:

\[
L = \mathrm{diag}(v(x_t, u_t, t)),
\]

so each state coordinate receives its own independent Brownian driver.

If `bm_dim = 1`, the same vector is interpreted as a column loading vector:

\[
L(x_t, u_t, t) = v(x_t, u_t, t) \in \mathbb{R}^{d_x \times 1},
\]

so all state coordinates share a single Brownian path.

### `ScalarDiffusion`

`ScalarDiffusion` represents a scalar-valued coefficient

\[
\sigma(x_t, u_t, t) \in \mathbb{R}.
\]

Its interpretation also depends on `bm_dim`.

If `bm_dim = d_x`, it is interpreted as isotropic independent noise:

\[
L = \sigma(x_t, u_t, t)\,I_{d_w}
\]

with \(d_w = d_x\).

If `bm_dim = 1`, it is interpreted as a shared scalar driver applied equally to
every state coordinate:

\[
L = \sigma(x_t, u_t, t)\,\mathbf{1}_{d_x},
\]

viewed as a column vector in \(\mathbb{R}^{d_x \times 1}\).

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

```python
import jax.numpy as jnp

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=ScalarDiffusion(lambda x, u, t: 0.1 + 0.05 * jnp.tanh(x[0]), bm_dim=2),
)
```

## API

::: dynestyx.models.diffusions.Diffusion
    options:
      show_root_heading: false
      filters:
        - "!^evaluate_value$"
        - "!^resolve_metadata$"
        - "!^evaluate$"
        - "!^as_matrix$"
        - "!^gram_matrix$"
        - "!^apply$"

### `FullDiffusion`

::: dynestyx.models.diffusions.FullDiffusion
    options:
      show_root_heading: false
      filters:
        - "!^evaluate_value$"
        - "!^resolve_metadata$"
        - "!^evaluate$"
        - "!^as_matrix$"
        - "!^gram_matrix$"
        - "!^apply$"

### `DiagonalDiffusion`

::: dynestyx.models.diffusions.DiagonalDiffusion
    options:
      show_root_heading: false
      filters:
        - "!^evaluate_value$"
        - "!^resolve_metadata$"
        - "!^evaluate$"
        - "!^as_matrix$"
        - "!^gram_matrix$"
        - "!^apply$"

### `ScalarDiffusion`

::: dynestyx.models.diffusions.ScalarDiffusion
    options:
      show_root_heading: false
      filters:
        - "!^evaluate_value$"
        - "!^resolve_metadata$"
        - "!^evaluate$"
        - "!^as_matrix$"
        - "!^gram_matrix$"
        - "!^apply$"
