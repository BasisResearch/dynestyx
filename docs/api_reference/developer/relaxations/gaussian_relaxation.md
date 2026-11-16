# Gaussian relaxation

`GaussianRelaxation` transforms selected model components before inference or
simulation. `relax_dynamics` applies the same transformation directly.
`state_evolution_cov` can only be applied to discrete-time dynamics.
See the [public guide](../../public/relaxations/gaussian_relaxation.md) for examples,
the [model reference](../../public/models/specialized_models.md#covariance-specifications)
for covariance specifications, and [handlers](../../public/handlers.md#handler-order)
for nesting order.

::: dynestyx.relaxations
    options:
      filters: []
      show_root_heading: false
      show_root_toc_entry: false
