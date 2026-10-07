# Specialized Models

## Observation models

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

Scalar, diagonal, and full covariance objects are reusable in Gaussian models
and [Gaussian relaxation](../relaxations/gaussian_relaxation.md). Scalar and diagonal
objects require exactly one of `sd` or `variance`; full objects take a matrix.
All three preserve leading batch/plate axes as Equinox pytrees.

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
