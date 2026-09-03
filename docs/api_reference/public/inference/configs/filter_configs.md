# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`. We provide a summary below, as well as an exhaustive list of classes.

`include_predicted_observations` controls whether supported backend predictive-observation outputs are collected into `ConditionedResult` and defaults to `True`. The shared `record_predicted_observations_*` fields independently control whether available means, covariances, or ensembles are recorded to the NumPyro trace; they also default to `True`. Observation scoring is configured on the separate `Evaluation` handler.

## EnKF localization

`EnKFConfig.localization` accepts either `EnKFLocalizationConfig` for covariance tapering from pairwise distances or `EnKFLocalizationFunctions` for direct control of Cuthbert's localization callbacks. Localization is supported by the discrete Cuthbert EnKF. `ContinuousTimeEnKFConfig` rejects it; for deterministic continuous-time dynamics, combine `EnKFConfig` with an ODE-flow `Discretizer` instead.

The built-in taper choices are `"gaspari_cohn"` and `"gaussian"`. Supplying `observation_distances` localizes both the state–observation cross covariance and the observation marginal covariance; omitting it localizes only the cross covariance. A callable taper receives a distance matrix and may close over differentiable JAX parameters:

```python
import jax.numpy as jnp

from dynestyx.inference.filters import EnKFConfig, EnKFLocalizationConfig


def gaussian_taper(distances):
    length_scale = 3.0
    return jnp.exp(-0.5 * (distances / length_scale) ** 2)


localization = EnKFLocalizationConfig(
    state_observation_distances=cross_distances,
    observation_distances=observation_distances,
    taper=gaussian_taper,
)
filter_config = EnKFConfig(localization=localization)
```

Advanced users may instead provide Cuthbert-level callbacks. A custom marginal innovation constructor must be paired with a predictive covariance modifier so filtering likelihoods and observation scores use the same covariance:

```python
from cuthbertlib.ensemble_kalman import construct_tapered_chol_innovation_covariance
from dynestyx.inference.filters import EnKFConfig, EnKFLocalizationFunctions


def modify_cross_covariance(cross_covariance, model_inputs):
    return cross_taper * cross_covariance


def construct_chol_innovation_covariance(Y, chol_R, model_inputs):
    return construct_tapered_chol_innovation_covariance(Y, chol_observation_taper, chol_R)


def modify_predicted_observation_covariance(covariance, model_inputs):
    return observation_taper * covariance


filter_config = EnKFConfig(
    localization=EnKFLocalizationFunctions(
        modify_cross_covariance=modify_cross_covariance,
        construct_chol_innovation_covariance=construct_chol_innovation_covariance,
        modify_predicted_observation_covariance=modify_predicted_observation_covariance,
    )
)
```

Projected forecast ensembles remain the raw ensemble even when the observation marginal is localized. For an `EnergyScore` intended to represent the localized Gaussian covariance, select `ObservationScoringConfig(sample_source="gaussian_moments")`.

## Available filter configurations

| Config class               | Time domain         | When it fits best |
|----------------------------|---------------------|-------------------|
| `KFConfig`                 | Discrete            | Linear-Gaussian dynamics and linear-Gaussian observations (exact & optimal). |
| `EnKFConfig`               | Discrete            | Nonlinear or expensive models with Gaussian observations; cuthbert-backed and a good general-purpose default. *(default)* |
| `EKFConfig`                | Discrete            | Nonlinear, differentiable Gaussian dynamics, nonlinear (and with `cuthbert`, non-Gaussian) but differentiable observations (approximate). |
| `UKFConfig`                | Discrete            | Nonlinear, differentiable Gaussian dynamics, nonlinear but differentiable Gaussian observations (approximate). Generally more accurate, but slower than `EKFConfig`. |
| `PFConfig`                 | Discrete            | Applicable for arbitrary state-space models, but quite expensive and noisy estimates (asymptotically exact in the limit of infinite particles, approximate in practice). |
| `HMMConfig`                | Discrete (HMM)      | Finite discrete latent state space (exact & optimal). |
| `ContinuousTimeKFConfig`   | Continuous-discrete | Linear-Gaussian SDE + linear-Gaussian observations (exact and optimal). |
| `ContinuousTimeEKFConfig`  | Continuous-discrete | Mildly nonlinear SDE with differentiable drift and difussion terms; Gaussian observations (approximate). |
| `ContinuousTimeUKFConfig`  | Continuous-discrete | Nonlinear SDE; derivative-free; Gaussian observations (approximate). Generally more accurate, but slower than `ContinuousTimeEKFConfig`. |
| `ContinuousTimeEnKFConfig` | Continuous-discrete | High-dimensional or expensive models with lower-dimensional structure and Gaussian observations (approximate). Performs reasonably as a default. *(default)* |
| `ContinuousTimeDPFConfig`  | Continuous-discrete | Applicable for arbitrary state-space models, but quite expensive and noisy estimates (asymptotically exact in the limit of infinite particles, approximate in practice). |

## Discrete Time Configuration Classes

::: dynestyx.inference.configs.filter
    options:
      members:
        - BaseFilterConfig
        - KFConfig
        - EKFConfig
        - UKFConfig
        - PFConfig
        - TaperCovarianceFn
        - ModifyCrossCovariance
        - ConstructCholInnovationCovariance
        - ModifyPredictedObservationCovariance
        - EnKFLocalizationConfig
        - EnKFLocalizationFunctions
        - EnKFConfig

## Continuous Time Configuration Classes

::: dynestyx.inference.configs.filter
    options:
      members:
        - ContinuousTimeConfig
        - ContinuousTimeKFConfig
        - ContinuousTimeEKFConfig
        - ContinuousTimeKFConfig
        - ContinuousTimePFConfig
        - ContinuousTimeEnKFConfig


## Discrete State-Space Configuration Classes

::: dynestyx.inference.configs.filter
    options:
      members:
        - HMMConfig
