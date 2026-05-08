# Smoother Configurations

Developer-facing API reference for smoother config classes and dispatch types.

Concrete smoother configs intentionally mirror the filter config hierarchy while
requiring users to opt into smoothing-specific classes. Backend support is
validated by `Smoother` before dispatch:

- discrete `KFSmootherConfig` and `EKFSmootherConfig`: `cuthbert` or `cd_dynamax`
- discrete `UKFSmootherConfig`: `cd_dynamax`
- discrete `PFSmootherConfig`: `cuthbert`
- continuous `ContinuousTimeKFSmootherConfig` and
  `ContinuousTimeEKFSmootherConfig`: `cd_dynamax`

Smoother-specific fields live on the concrete classes rather than a nested
options object, which keeps handler dispatch and API docs aligned.

::: dynestyx.inference.smoother_configs
    options:
      filters: []
