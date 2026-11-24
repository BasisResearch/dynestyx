# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`. We provide a summary below, as well as an exhaustive list of classes.

Filter configs, including localization and resampling options, are frozen dataclasses.
Create an updated config with `dataclasses.replace` instead of assigning to its fields:

```python
from dataclasses import replace

import dynestyx as dsx

config = dsx.configs.EnKFConfig(n_particles=30)
larger_ensemble = replace(config, n_particles=100)
```

Replacement reruns configuration validation and leaves the original unchanged.
Nested objects such as `extra_filter_kwargs` dictionaries are not recursively frozen.

`include_predicted_observations` controls whether supported backend predictive-observation outputs are collected into `ConditionedResult` and defaults to `True`. The shared `record_predicted_observations_*` fields independently control whether available means, covariances, or ensembles are recorded to the NumPyro trace; they also default to `True`. Observation scoring is configured on the separate `Evaluation` handler.

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

## Custom filters

Dynestyx also allows the use of custom filters, conforming to a “predict/update” interface (a `SplitFilter`), or a “combined” interface (corresponding roughly to how `cuthbert` filters are implemented). Pass `CustomFilterConfig(build_filter=..., gather_outputs=...)` to `dsx.Filter`. The `build_filter` argument provides a factory that takes a `DynamicalModel` and returns the corresponding filter object.

### Split filters

Return a `SplitFilter(initialize, predict, update)` from `dynestyx.inference.custom_filter`. Write $q_k^-(x)$ for the predictive belief about $x_k$ before observing $y_k$, and $q_k(x)$ for the filtered belief after observing it. Each callback works with your chosen representation of these beliefs, such as a Gaussian mean and covariance or a set of weighted particles. Annotate the factory return type as `SplitFilter[Prior, Filtered]` using the types returned by `predict` and `update`, respectively. The corresponding gather callback has signature `(Prior, Filtered) -> FilterOutputs`.

1. `initialize(inputs, *, key)` returns the prior $q_1^-(x)=p(x_1=x)$ at the first observation time, with cumulative log likelihood initialized to zero. Dynestyx passes this prior directly to `update` for the first observation.
2. `predict(filtered, inputs, *, key)` propagates the previous filtered belief through the model transition. For $k>1$, return a representation or approximation of $q_k^-(x)=\int p(x_k=x\mid x_{k-1}=z,u_{k-1})\,q_{k-1}(z)\,dz$. The interval endpoints are `inputs.time_prev` and `inputs.time`, and its control is `inputs.u_prev`.
3. `update(prior, inputs, *, key)` conditions the predictive belief on `inputs.y`. With observation likelihood $g_k(x)=p(y_k\mid x_k=x,u_k)$, the predictive probability or density of the observation is $Z_k=\int g_k(x)\,q_k^-(x)\,dx$. Return a filtered state representing $q_k(x)=g_k(x)\,q_k^-(x)/Z_k$ and carry forward the cumulative log likelihood $\ell_k=\ell_{k-1}+\log Z_k$. Approximate filters use their own approximations to these integrals.

### Combined filters

A combined filter implements `init_prepare(*, key)`, `filter_prepare(inputs, *, key)`, and `filter_combine(previous, prepared)`. Initialization returns a state $s_0$ encoding the initial prior and $\ell_0=0$. At observation $k$, preparation builds an element $e_k$ using only the current step inputs and its key. For example, $e_k$ can hold the observation and the parameters needed to evaluate the transition and observation likelihood. The combine operation computes $s_k=s_{k-1}\otimes e_k$, which must encode the filtered belief $q_k$ and cumulative log likelihood $\ell_k$ defined above. Use `inputs.is_first_step` to assimilate $y_1$ directly into the initial prior.

For parallel filtering, each prepared element must also be a valid operand for combining adjacent observation blocks. Combining a block for $i$ through $j$ with one for $j+1$ through $k$ must summarize the block for $i$ through $k$, with $(a\otimes b)\otimes c=a\otimes(b\otimes c)$. A prefix $s_0\otimes e_1\otimes\cdots\otimes e_k$ then gives the same filtered state regardless of grouping. Set `associative=True` on the filter object and `parallel=True` on the config to use this execution mode.

### Gathering outputs

Implement `gather_outputs(prior, filtered)` for a split filter or `gather_outputs(filtered)` for a combined filter. Return `FilterOutputs` with `log_normalizing_constant` set to $\ell_k=\sum_{j=1}^k\log Z_j$ and either the mean and covariance of $q_k$ or its particles and log weights. The last $\ell_k$ becomes `result.marginal_loglik`. Optional `predicted_observations` describe $p(y_k\mid y_{1:k-1})$ before the update and support observation scoring. A fixed dictionary of diagnostic arrays becomes `result.extra_outputs`, with leading plate dimensions followed by observation time.

[Build a custom filter with Dynestyx](../../../deep_dives/custom_gauss_hermite_filter.ipynb)

## EnKF localization

Localization is a common technique in EnKFs, related to the issue of estimating a covariance matrix from a few samples. The (cuthbert, discrete-time) EnKF supports localization via covariance tapering via pairwise distances, or direct control over localization functions, using `EnKFLocalizationConfig` for the former and `EnKFLocalizationFunctions` for the latter. For localization in a deterministic continuous-time dynamics, combine `EnKFConfig` with an ODE-flow `Discretizer`.

For a worked example with a custom differentiable taper, see [Learning EnKF localization with proper scoring rules](../../../deep_dives/l96_localization_hyperparameter_scoring.ipynb). For the mathematical definitions of the taper functions and the factorization of the localized innovation covariance, see [Cuthbert's covariance localization reference](https://state-space-models.github.io/cuthbert/api_cuthbertlib/ensemble_kalman/localization/).

Dynestyx offers a few built-in taper functions for commonly-used correlation functions, namely `"gaspari_cohn"` and `"gaussian"`. Supplying `observation_distances` localizes both the state–observation cross covariance and the observation marginal covariance; omitting it localizes only the cross covariance. A callable taper receives a distance matrix and may close over differentiable JAX parameters:

```python
import jax.numpy as jnp

import dynestyx as dsx


def gaussian_taper(distances):
    length_scale = 3.0
    return jnp.exp(-0.5 * (distances / length_scale) ** 2)


localization = dsx.configs.EnKFLocalizationConfig(
    state_observation_distances=cross_distances,
    observation_distances=observation_distances,
    taper=gaussian_taper,
)
filter_config = dsx.configs.EnKFConfig(localization=localization)
```

For advanced usage, one may provide directly the functions that modify the covariance and cross-covariance. [Cuthbert's localization tutorial](https://state-space-models.github.io/cuthbert/examples/enkf_localization_l96/) shows how to construct these callbacks from distance-based tapers and explains their role in the EnKF update:

```python
from cuthbertlib.ensemble_kalman import construct_tapered_chol_innovation_covariance

import dynestyx as dsx


def modify_cross_covariance(cross_covariance, model_inputs):
    return cross_taper * cross_covariance


def construct_chol_innovation_covariance(Y, chol_R, model_inputs):
    return construct_tapered_chol_innovation_covariance(Y, chol_observation_taper, chol_R)


def modify_predicted_observation_covariance(covariance, model_inputs):
    return observation_taper * covariance


filter_config = dsx.configs.EnKFConfig(
    localization=dsx.configs.EnKFLocalizationFunctions(
        modify_cross_covariance=modify_cross_covariance,
        construct_chol_innovation_covariance=construct_chol_innovation_covariance,
        modify_predicted_observation_covariance=modify_predicted_observation_covariance,
    )
)
```

Projected forecast ensembles remain the raw ensemble even when the observation marginal is localized. For an `EnergyScore` intended to represent the localized Gaussian covariance, select `dsx.configs.ObservationScoringConfig(sample_source="gaussian_moments")`.

## Discrete Time Configuration Classes

::: dynestyx.configs.filters
    options:
      members:
        - BaseFilterConfig
        - KFConfig
        - EKFConfig
        - UKFConfig
        - PFConfig
        - PFResamplingConfig
        - TaperCovarianceFn
        - ModifyCrossCovariance
        - ConstructCholInnovationCovariance
        - ModifyPredictedObservationCovariance
        - EnKFLocalizationConfig
        - EnKFLocalizationFunctions
        - EnKFConfig

## Continuous Time Configuration Classes

::: dynestyx.configs.filters
    options:
      members:
        - ContinuousTimeConfig
        - ContinuousTimeKFConfig
        - ContinuousTimeEKFConfig
        - ContinuousTimeUKFConfig
        - ContinuousTimeDPFConfig
        - ContinuousTimeEnKFConfig


## Discrete State-Space Configuration Classes

::: dynestyx.configs.filters
    options:
      members:
        - HMMConfig
