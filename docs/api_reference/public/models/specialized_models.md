# Specialized Models

## Observation models

For exact identity measurements (`y = x`), use `DiracIdentityObservation()`.
With `LatentPathBuilder`, observed values fix those state coordinates, and only
missing state coordinates are inferred. [Relaxing the observation](../relaxations/gaussian_relaxation.md)
turns these into noisy Gaussian measurements: observed values contribute a
likelihood, and the latent state coordinates are inferred.

::: dynestyx.models.observations
    options:
      members:
        - LinearGaussianObservation
        - GaussianObservation
        - DeterministicObservation
        - DiracIdentityObservation

## State evolution models

::: dynestyx.models.state_evolution
    options:
      members:
        - LinearGaussianStateEvolution
        - GaussianStateEvolution
        - DeterministicStateEvolution

## Covariance specifications

Scalar, diagonal, and full covariance objects are reusable in Gaussian models,
`LTI_discrete`, and [Gaussian relaxation](../relaxations/gaussian_relaxation.md):

- `ScalarCovariance(sd=... | variance=...)` represents isotropic covariance.
- `DiagonalCovariance(sd=... | variance=...)` holds one variance per event coordinate.
- `FullCovariance(matrix)` holds a dense covariance matrix.

Scalar and diagonal objects require exactly one of `sd` or `variance`. All three
preserve leading batch/plate axes as Equinox pytrees; their trailing covariance
event axes have rank zero, one, or two respectively.

Relaxation settings also accept shorthand: a scalar is a **variance**, a vector
contains **diagonal variances**, and a matrix is a **full covariance**. Use
explicit covariance objects for batched settings to distinguish plate axes
from event axes.

Covariance objects check shape, finite values, and symmetry. Supply
positive-semidefinite covariances and a positive-definite effective covariance
for Gaussian densities. Zero covariance can be added to an already
positive-definite Gaussian. Relaxation uses the specified covariance exactly;
the EnKF setting `recorded_filtered_states_cov_jitter` only adds jitter to returned
state distributions, leaving the filter recursion and marginal likelihood
unchanged.

::: dynestyx.models.covariances
    options:
        members:
            - Covariance
            - ScalarCovariance
            - DiagonalCovariance
            - FullCovariance

## LTI model factories

::: dynestyx.models.lti_dynamics
    options:
      members:
        - LTI_continuous
        - LTI_discrete
