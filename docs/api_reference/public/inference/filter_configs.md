# Filter Configurations

The single `Filter()` handler is directed to the appropriate filtering algorithm via the provided `FilterConfig`. We provide a summary below, as well as an exhaustive list of classes.

## Available filter configurations

| Config class               | Time domain         | When it fits best |
|----------------------------|---------------------|-------------------|
| `KFConfig`                 | Discrete            | Linear-Gaussian dynamics and linear-Gaussian observations (exact & optimal). |
| `EnKFConfig`               | Discrete            | Nonlinear or expensive models with Gaussian observations; cuthbert-backed and a good general-purpose default. *(default)* |
| `EKFConfig`                | Discrete            | Nonlinear, differentiable Gaussian dynamics, nonlinear but differentiable Gaussian observations (approximate). |
| `UKFConfig`                | Discrete            | Nonlinear, differentiable Gaussian dynamics, nonlinear but differentiable Gaussian observations (approximate). Generally more accurate, but slower than `EKFConfig`. |
| `PFConfig`                 | Discrete            | Applicable for arbitrary state-space models, but quite expensive and noisy estimates (asymptotically exact in the limit of infinite particles, approximate in practice). |
| `HMMConfig`                | Discrete (HMM)      | Finite discrete latent state space (exact & optimal). |
| `ContinuousTimeKFConfig`   | Continuous-discrete | Linear-Gaussian SDE + linear-Gaussian observations (exact and optimal). |
| `ContinuousTimeEKFConfig`  | Continuous-discrete | Mildly nonlinear SDE with differentiable drift and difussion terms; Gaussian observations (approximate). |
| `ContinuousTimeUKFConfig`  | Continuous-discrete | Nonlinear SDE; derivative-free; Gaussian observations (approximate). Generally more accurate, but slower than `ContinuousTimeEKFConfig`. |
| `ContinuousTimeEnKFConfig` | Continuous-discrete | High-dimensional or expensive models with lower-dimensional structure and Gaussian observations (approximate). Performs reasonably as a default. *(default)* |
| `ContinuousTimeDPFConfig`  | Continuous-discrete | Applicable for arbitrary state-space models, but quite expensive and noisy estimates (asymptotically exact in the limit of infinite particles, approximate in practice). |

## Discrete Time Configuration Classes

::: dynestyx.inference.filter_configs
    options:
      members:
        - BaseFilterConfig
        - KFConfig
        - EKFConfig
        - UKFConfig
        - PFConfig
        - EnKFConfig

## Continuous Time Configuration Classes

::: dynestyx.inference.filter_configs
    options:
      members:
        - ContinuousTimeConfig
        - ContinuousTimeKFConfig
        - ContinuousTimeEKFConfig
        - ContinuousTimeKFConfig
        - ContinuousTimePFConfig
        - ContinuousTimeEnKFConfig


## Discrete State-Space Configuration Classes

::: dynestyx.inference.filter_configs
    options:
      members:
        - HMMConfig
