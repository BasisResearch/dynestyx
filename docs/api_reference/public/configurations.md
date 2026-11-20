# Configurations

Use `dynestyx.configs` to find the configuration objects for filters, smoothers,
samplers, discretizers, simulators, and observation scoring.

```python
import dynestyx as dsx

filter_handler = dsx.Filter(filter_config=dsx.configs.EnKFConfig(n_particles=64))
```

## Filters

Pass a filter config to `dsx.Filter(filter_config=...)`.
See [Filter configurations](inference/configs/filter_configs.md) for algorithm and backend choices.

### Discrete-time filters

- `dsx.configs.KFConfig`
- `dsx.configs.EKFConfig`
- `dsx.configs.UKFConfig`
- `dsx.configs.EnKFConfig`
- `dsx.configs.PFConfig`

### Continuous-time filters

These filters handle continuous-time dynamics with discrete observations.

- `dsx.configs.ContinuousTimeKFConfig`
- `dsx.configs.ContinuousTimeEKFConfig`
- `dsx.configs.ContinuousTimeUKFConfig`
- `dsx.configs.ContinuousTimeEnKFConfig`
- `dsx.configs.ContinuousTimeDPFConfig`

### Finite-state filters

- `dsx.configs.HMMConfig`

### Localization and resampling

- `dsx.configs.EnKFLocalizationConfig` for distance-based covariance tapering via `dsx.configs.EnKFConfig(localization=...)`
- `dsx.configs.EnKFLocalizationFunctions` for custom localization callbacks via `dsx.configs.EnKFConfig(localization=...)`
- `dsx.configs.PFResamplingConfig` for resampling options via `dsx.configs.PFConfig(resampling_method=...)`

## Smoothers

Pass a smoother config to `dsx.Smoother(smoother_config=...)`.
See [Smoother configurations](inference/configs/smoother_configs.md) for supported backends and options.

- `dsx.configs.KFSmootherConfig`
- `dsx.configs.EKFSmootherConfig`
- `dsx.configs.UKFSmootherConfig`
- `dsx.configs.EnRTSSmootherConfig`
- `dsx.configs.PFSmootherConfig`
- `dsx.configs.ContinuousTimeKFSmootherConfig`
- `dsx.configs.ContinuousTimeEKFSmootherConfig`

## MCMC

Pass a sampler config to `MCMCInference(mcmc_config=..., model=...)`.
See [MCMC configurations](inference/configs/mcmc_configs.md) for sampler and backend options.

- `dsx.configs.NUTSConfig`
- `dsx.configs.HMCConfig`
- `dsx.configs.AdaptiveMetropolisConfig`
- `dsx.configs.AdaptiveMWGConfig`
- `dsx.configs.SGLDConfig`
- `dsx.configs.MALAConfig`

## Discretizers

Pass a discretizer config to `dsx.Discretizer(discretizer_config=...)`.
See [Discretizer configurations](inference/configs/discretizer_configs.md) for method selection and the `dsx.configs.DiscretizerConfig` type alias.

- `dsx.configs.ODEFlowConfig`
- `dsx.configs.EulerMaruyamaConfig`
- `dsx.configs.ExactAffineConfig`
- `dsx.configs.LocalLinearizationConfig`
- `dsx.configs.MeanTrajectoryLinearizationConfig`
- `dsx.configs.DiffraxSampleConfig`

## Simulators

Pass a simulator config to `dsx.Simulator(simulator_config=...)`.
See [Simulator configurations](simulators/simulator_configs.md) for solver settings and the `dsx.configs.SimulatorConfig` type alias.

- `dsx.configs.ODESimulatorConfig`
- `dsx.configs.SDESimulatorConfig`

## Observation scoring

Pass `dsx.configs.ObservationScoringConfig` to `dsx.Evaluation(observation_scoring_config=...)`.
See [Scoring configs](inference/configs/scoring_configs.md) for score and predictive sample options.
