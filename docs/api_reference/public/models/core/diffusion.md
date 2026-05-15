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

- If your diffusion has the form \(L = \sigma I_{d_w}\) with \(d_w = d_x\), use `ScalarDiffusion(sigma, bm_dim=state_dim)`.
- If your diffusion has the form \(L = \sigma \mathbf{1}_{d_x}\), use `ScalarDiffusion(sigma, bm_dim=1)`.
- If your diffusion has the form \(L = \mathrm{diag}(v)\), use `DiagonalDiffusion(v, bm_dim=state_dim)`.
- If your diffusion has the form \(L = v \in \mathbb{R}^{d_x \times 1}\), use `DiagonalDiffusion(v, bm_dim=1)`.
- If your diffusion is a general matrix \(L \in \mathbb{R}^{d_x \times d_w}\), use `FullDiffusion(L)`.

The same constructors also accept callables `(x, u, t) -> value` instead of
constants. For example, if \(L(x_t, u_t, t) = \sigma(x_t) I_{d_w}\), use
`ScalarDiffusion(lambda x, u, t: sigma(x), bm_dim=state_dim)`. The same pattern
applies across `ScalarDiffusion`, `DiagonalDiffusion`, and `FullDiffusion`.

In practice, many models use a constant diffusion coefficient. In those cases,
pass the matrix/vector/scalar value directly. Reserve callable diffusion
coefficients for cases where the coefficient genuinely depends on state,
control, or time.

## `Diffusion`

`Diffusion` is the common base class for structured diffusion coefficients.
Most users should instantiate one of its public subclasses instead of using
`Diffusion` directly:

- Use `ScalarDiffusion` when one scalar scale is enough. This is the most common choice for isotropic diffusion.
- Use `DiagonalDiffusion` when each state coordinate should have its own scale but the loading remains axis-aligned.
- Use `FullDiffusion` when you need to specify a genuinely matrix-valued loading.

Each class accepts either:

- a constant coefficient, or
- a callable `(x, u, t) -> value`.

### `FullDiffusion`

Mathematically, `FullDiffusion` represents
\(L(x_t, u_t, t) \in \mathbb{R}^{d_x \times d_w}\).

Use `FullDiffusion(coefficient, bm_dim=None)` when you want to specify this
matrix-valued diffusion coefficient directly.

Accepted `coefficient` forms:

- a constant array with trailing shape `(state_dim, bm_dim)`, or
- a callable `(x, u, t) -> array` with trailing shape `(state_dim, bm_dim)`.

If `coefficient` is constant, `bm_dim` is inferred automatically when omitted.

Example:

```python
import jax.numpy as jnp
from dynestyx import ContinuousTimeStateEvolution, FullDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=FullDiffusion(jnp.eye(2)),
)
```

### `DiagonalDiffusion`

Mathematically, `DiagonalDiffusion` represents a vector-valued coefficient
\(v(x_t, u_t, t) \in \mathbb{R}^{d_x}\). If `bm_dim = d_x`, this means
\(L = \mathrm{diag}(v(x_t, u_t, t))\). If `bm_dim = 1`, this means
\(L = v(x_t, u_t, t) \in \mathbb{R}^{d_x \times 1}\).

Use `DiagonalDiffusion(coefficient, bm_dim)` when the diffusion is naturally
parameterized by a vector of state-wise loadings.

Accepted `coefficient` forms:

- a constant vector with trailing shape `(state_dim,)`, or
- a callable `(x, u, t) -> array` with trailing shape `(state_dim,)`.

`bm_dim` is required and must be either `1` or `state_dim`.

Example:

```python
import jax.numpy as jnp
from dynestyx import ContinuousTimeStateEvolution, DiagonalDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=DiagonalDiffusion(jnp.array([0.1, 0.2]), bm_dim=2),
)
```

### `ScalarDiffusion`

Mathematically, `ScalarDiffusion` represents a scalar-valued coefficient
\(\sigma(x_t, u_t, t) \in \mathbb{R}\). If `bm_dim = d_x`, this means
\(L = \sigma(x_t, u_t, t)\,I_{d_w}\) with \(d_w = d_x\). If `bm_dim = 1`, this
means \(L = \sigma(x_t, u_t, t)\,\mathbf{1}_{d_x}\), viewed as a column vector
in \(\mathbb{R}^{d_x \times 1}\).

Use `ScalarDiffusion(coefficient, bm_dim)` when one scalar scale is enough.
This is usually the simplest choice for isotropic diffusion.

Accepted `coefficient` forms:

- a scalar,
- a constant array with trailing shape `(1,)`, or
- a callable `(x, u, t) -> scalar_or_length_1_array`.

`bm_dim` is required and must be either `1` or `state_dim`.

Constant example:

```python
from dynestyx import ContinuousTimeStateEvolution, ScalarDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=ScalarDiffusion(0.1, bm_dim=2),
)
```

Callable example:

```python
import jax.numpy as jnp
from dynestyx import ContinuousTimeStateEvolution, ScalarDiffusion

state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: -x,
    diffusion=ScalarDiffusion(
        lambda x, u, t: 0.1 + 0.05 * jnp.tanh(x[0]),
        bm_dim=2,
    ),
)
```

## API

### `Diffusion`

::: dynestyx.models.diffusions.Diffusion
    options:
      show_root_heading: false
      show_root_toc_entry: false
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
      show_root_toc_entry: false
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
      show_root_toc_entry: false
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
      show_root_toc_entry: false
      filters:
        - "!^evaluate_value$"
        - "!^resolve_metadata$"
        - "!^evaluate$"
        - "!^as_matrix$"
        - "!^gram_matrix$"
        - "!^apply$"
