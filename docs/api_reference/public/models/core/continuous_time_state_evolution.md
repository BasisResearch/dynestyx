# ContinuousTimeStateEvolution

`ContinuousTimeStateEvolution` is the public entry point for defining continuous-time
state evolution. Most users should instantiate this class directly and pass an optional
`diffusion=` built from [`FullDiffusion`](./diffusion.md),
[`DiagonalDiffusion`](./diffusion.md), or [`ScalarDiffusion`](./diffusion.md).

Internally, `DynamicalModel` refines continuous-time dynamics to
deterministic and stochastic subclasses. Those specialized classes are intended for
developer-facing integrations and are documented in the developer API rather than the
public tutorials.

::: dynestyx.models.core.ContinuousTimeStateEvolution
    options:
      show_root_heading: false
      show_root_toc_entry: false
