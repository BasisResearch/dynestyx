# Diffusions

This page documents the diffusion internals used by continuous-time solvers,
discretizers, and backend integrations.

## Overview

Public model code should usually work with `FullDiffusion`,
`DiagonalDiffusion`, and `ScalarDiffusion` through
`ContinuousTimeStateEvolution(diffusion=...)`.

Developer-facing code sometimes needs the lower-level evaluation layer:

- `Diffusion.evaluate(...)` evaluates a structured diffusion object at a
  concrete `(x, u, t)`.
- `EvaluatedDiffusion.as_matrix(...)` returns the corresponding matrix `L`.
- `EvaluatedDiffusion.apply(...)` applies that matrix to a Brownian increment.
- `resolve_metadata(...)` canonicalizes `bm_dim` and validates the coefficient
  shape against `state_dim`.

In other words, the public classes describe the structure of the diffusion,
while `EvaluatedDiffusion` is the solver-facing object used after a concrete
state, control, and time have been chosen.

## API

### `EvaluatedDiffusion`

::: dynestyx.models.diffusions.EvaluatedDiffusion
    options:
      show_root_heading: false

### `Diffusion`

::: dynestyx.models.diffusions.Diffusion
    options:
      show_root_heading: false
      filters: []

### `FullDiffusion`

::: dynestyx.models.diffusions.FullDiffusion
    options:
      show_root_heading: false
      filters: []

### `DiagonalDiffusion`

::: dynestyx.models.diffusions.DiagonalDiffusion
    options:
      show_root_heading: false
      filters: []

### `ScalarDiffusion`

::: dynestyx.models.diffusions.ScalarDiffusion
    options:
      show_root_heading: false
      filters: []
