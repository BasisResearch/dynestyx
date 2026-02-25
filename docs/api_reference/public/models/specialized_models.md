# Specialized Models

## Observation models

::: dynestyx.models.observations
    options:
      members:
        - LinearGaussianObservation
        - GaussianObservation
        - DiracIdentityObservation

## State evolution models

::: dynestyx.models.state_evolution
    options:
      members:
        - LinearGaussianStateEvolution
        - GaussianStateEvolution
        - AffineDrift

## LTI model factories

::: dynestyx.models.lti_dynamics
    options:
      members:
        - LTI_continuous
        - LTI_discrete

