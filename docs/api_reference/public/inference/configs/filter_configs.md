# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`. We provide a summary below, as well as an exhaustive list of classes.

Filter configs, including localization and resampling options, are frozen dataclasses.
Create an updated config with `dataclasses.replace` instead of assigning to its fields:

```python
from dataclasses import replace

from dynestyx.inference.configs.filter import EnKFConfig

config = EnKFConfig(n_particles=30)
larger_ensemble = replace(config, n_particles=100)
```

Replacement reruns configuration validation and leaves the original unchanged.
Nested objects such as `extra_filter_kwargs` dictionaries are not recursively frozen.

`include_predicted_observations` controls whether supported backend predictive-observation outputs are collected into `ConditionedResult` and defaults to `True`. The shared `record_predicted_observations_*` fields independently control whether available means, covariances, or ensembles are recorded to the NumPyro trace; they also default to `True`. Observation scoring is configured on the separate `Evaluation` handler.

## Available filter configurations

| Config class               | Time domain         | When it fits best |
|----------------------------|---------------------|-------------------|
| `KFConfig`                 | Discrete            | Linear-Gaussian dynamics and linear-Gaussian observations (exact & optimal). |
| `LETKFConfig`              | Discrete            | Deterministic local ensemble transforms with explicit observation neighborhoods and diagonal Gaussian noise. |
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

## EnKF localization

Localization is a common technique in EnKFs, related to the issue of estimating a covariance matrix from a few samples. The (cuthbert, discrete-time) EnKF supports localization via covariance tapering via pairwise distances, or direct control over localization functions, using `EnKFLocalizationConfig` for the former and `EnKFLocalizationFunctions` for the latter. For localization in a deterministic continuous-time dynamics, combine `EnKFConfig` with an ODE-flow `Discretizer`.

For a worked example with a custom differentiable taper, see [Learning EnKF localization with proper scoring rules](../../../../deep_dives/l96_localization_hyperparameter_scoring.ipynb). For the mathematical definitions of the taper functions and the factorization of the localized innovation covariance, see [Cuthbert's covariance localization reference](https://state-space-models.github.io/cuthbert/api_cuthbertlib/ensemble_kalman/localization/).

Dynestyx offers a few built-in taper functions for commonly-used correlation functions, namely `"gaspari_cohn"` and `"gaussian"`. Supplying `observation_distances` localizes both the state–observation cross covariance and the observation marginal covariance; omitting it localizes only the cross covariance. A callable taper receives a distance matrix and may close over differentiable JAX parameters:

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

For advanced usage, one may provide directly the functions that modify the covariance and cross-covariance. [Cuthbert's localization tutorial](https://state-space-models.github.io/cuthbert/examples/enkf_localization_l96/) shows how to construct these callbacks from distance-based tapers and explains their role in the EnKF update:

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

## Local ensemble transform filtering (LETKF)

`LETKFConfig` selects Cuthbert's deterministic local ensemble transform Kalman
filter. Supply observation indices and precision weights, both shaped
`(state_dim, max_local_observations)`:

```python
import jax.numpy as jnp
import dynestyx as dsx
from dynestyx.inference.filters import Filter, LETKFConfig

# A two-coordinate state and two-dimensional observation.
config = LETKFConfig(
    n_particles=30,
    observation_indices=jnp.array([[0, -1], [0, 1]]),
    localization_weights=jnp.array([[1.0, 0.0], [0.3, 1.0]]),
    inflation_delta=0.05,
)
with Filter(filter_config=config):
    result = dsx.condition(
        "state", dynamics, obs_times=obs_times, obs_values=obs_values
    )
```

Indices refer to the original observation order. Use `-1` for padding; other
indices must be in range and unique within each row. Nonpadding weights must
be finite and nonnegative; zero excludes an observation. Padding weights are
ignored. Existing Cuthbert Gaussian and Gaspari–Cohn taper utilities can compute
precision weights from local distances.

For time-varying neighborhoods, supply `get_localization(model_inputs)` instead
of the two arrays. The callback receives the current `FilterModelInputs`,
including scalar `time` and observation-aligned controls `u`, and returns
`(observation_indices, localization_weights)`. Shapes must stay fixed across
time and JIT execution; indices and weights may vary and weights may depend on
differentiable parameters. Neighborhood validation also runs under JIT.

LETKF requires at least two ensemble members and state-independent diagonal
Gaussian observation noise. Scalar and diagonal covariance specifications,
including dense diagonal matrices and diagonal multivariate Gaussian
distributions, are accepted. Correlated observation noise is rejected. Noise
variances must be finite and positive in observed dimensions. NaN observations
are omitted from both analysis and likelihood. Empty or entirely excluded local
neighborhoods preserve their forecast members.

Forecast anomalies multiply by `1 + inflation_delta`, as in Cuthbert's EnKF.
The fixed random-key default, ensemble recording options, and covariance jitter
for returned distributions follow `EnKFConfig`. Forecast ensembles are retained
when `include_predicted_observations=True`.

The marginal log likelihood sums one **global Gaussian predictive density** per
time step using the full forecast observation covariance plus the original
observation noise. Each observed dimension is scored once, even when it occurs
in multiple neighborhoods or none. Localization changes analysis and later
forecasts; current weights do not directly alter the current predictive score.
An entirely missing observation contributes zero. This is a normalized
approximation, not an unbiased estimate of the original model likelihood.
Predictive observation summaries and evaluation scores use the global forecast.

LETKF is opt-in, uses sequential Cuthbert execution, and does not provide a
smoother. Discretize continuous-time models before using it. The backend is
pinned to [Cuthbert commit 2c4452a](https://github.com/DanWaxman/cuthbert/commit/2c4452a888e3b815c9666a94e0e1a907781d60fa).

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
        - LETKFConfig
        - GetLETKFLocalization

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
