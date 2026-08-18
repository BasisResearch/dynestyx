# Discretizer Configurations

Discretizer configs specify the discretization method used by a `Discretizer` handler. Available methods and information are specified below.

## Choosing a Method

| Configuration | Applicability | Transition interface | Typical cost per interval | Main tradeoff |
|---|---|---|---|---|
| `ODEFlowConfig` | Deterministic ODE | Delta or independent Normal density and samples | Numerical ODE solve | Accurate numerical flow; optional jitter changes the deterministic model |
| `EulerMaruyamaConfig` | General drift and diffusion | Gaussian density and samples | One drift and diffusion evaluation | Very cheap, inacurate over long discretization intervals |
| `ExactAffineConfig` | Affine drift, constant additive diffusion, no potential | Exact linear-Gaussian density and samples | Matrix exponentials | Only applicable to `AffineDrift` |
| `LocalLinearizationConfig` | Nonlinear drift, constant additive diffusion | Gaussian density and samples | Jacobian and matrix exponentials | Captures local stiffness; relinearizes at every state |
| `MeanTrajectoryLinearizationConfig` | General differentiable drift and diffusion | Gaussian density and samples | Joint mean/covariance ODE with Jacobians | Tracks the within-interval trajectory |
| `DiffraxSampleConfig` | General SDE supported by a Brownian-increment Diffrax solver | Samples only | Numerical SDE solve | Accurate simulation, but no transition density |

See [Comparing SDE discretization methods on Lorenz–63](../../../../deep_dives/sde_discretization_comparison.ipynb) for a worked comparison under a common EnKF and NUTS inference protocol.

## Configuration classes

::: dynestyx.inference.configs.discretizer
    options:
      members:
        - BaseDiscretizerConfig
        - DiscretizerConfig
        - ODEFlowConfig
        - EulerMaruyamaConfig
        - ExactAffineConfig
        - LocalLinearizationConfig
        - MeanTrajectoryLinearizationConfig
        - DiffraxSampleConfig
      show_root_heading: false
      show_root_toc_entry: false
