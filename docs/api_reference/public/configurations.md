# Configurations

Use `dynestyx.config` to find the configuration objects for filters, smoothers,
samplers, discretizers, simulators, and observation scoring.

```python
import dynestyx as dsx

filter_handler = dsx.Filter(filter_config=dsx.config.EnKFConfig(n_particles=64))
```

## Filters

Pass a filter config to `dsx.Filter(filter_config=...)`.
See [Filter configurations](inference/configs/filter_configs.md) for algorithm and backend choices.

### Discrete-time filters

- `dsx.config.KFConfig`
- `dsx.config.EKFConfig`
- `dsx.config.UKFConfig`
- `dsx.config.EnKFConfig`
- `dsx.config.PFConfig`

### Continuous-time filters

These filters handle continuous-time dynamics with discrete observations.

- `dsx.config.ContinuousTimeKFConfig`
- `dsx.config.ContinuousTimeEKFConfig`
- `dsx.config.ContinuousTimeUKFConfig`
- `dsx.config.ContinuousTimeEnKFConfig`
- `dsx.config.ContinuousTimeDPFConfig`

### Finite-state filters

- `dsx.config.HMMConfig`

### Localization and resampling

- `dsx.config.EnKFLocalizationConfig` for distance-based covariance tapering via `EnKFConfig(localization=...)`
- `dsx.config.EnKFLocalizationFunctions` for custom localization callbacks via `EnKFConfig(localization=...)`
- `dsx.config.PFResamplingConfig` for resampling options via `PFConfig(resampling_method=...)`

## Smoothers

Pass a smoother config to `dsx.Smoother(smoother_config=...)`.
See [Smoother configurations](inference/configs/smoother_configs.md) for supported backends and options.

- `dsx.config.KFSmootherConfig`
- `dsx.config.EKFSmootherConfig`
- `dsx.config.UKFSmootherConfig`
- `dsx.config.EnRTSSmootherConfig`
- `dsx.config.PFSmootherConfig`
- `dsx.config.ContinuousTimeKFSmootherConfig`
- `dsx.config.ContinuousTimeEKFSmootherConfig`

## MCMC

Pass a sampler config to `MCMCInference(mcmc_config=..., model=...)`.
See [MCMC configurations](inference/configs/mcmc_configs.md) for sampler and backend options.

- `dsx.config.NUTSConfig`
- `dsx.config.HMCConfig`
- `dsx.config.AdaptiveMetropolisConfig`
- `dsx.config.AdaptiveMWGConfig`
- `dsx.config.SGLDConfig`
- `dsx.config.MALAConfig`

## Discretizers

Pass a discretizer config to `dsx.Discretizer(discretizer_config=...)`.
See [Discretizer configurations](inference/configs/discretizer_configs.md) for method selection and the `dsx.config.DiscretizerConfig` type alias.

- `dsx.config.ODEFlowConfig`
- `dsx.config.EulerMaruyamaConfig`
- `dsx.config.ExactAffineConfig`
- `dsx.config.LocalLinearizationConfig`
- `dsx.config.MeanTrajectoryLinearizationConfig`
- `dsx.config.DiffraxSampleConfig`

## Simulators

Pass a simulator config to `dsx.Simulator(simulator_config=...)`.
See [Simulator configurations](simulators/simulator_configs.md) for solver settings and the `dsx.config.SimulatorConfig` type alias.

- `dsx.config.ODESimulatorConfig`
- `dsx.config.SDESimulatorConfig`

## Observation scoring

Pass `dsx.config.ObservationScoringConfig` to `dsx.Evaluation(observation_scoring_config=...)`.
See [Scoring configs](inference/configs/scoring_configs.md) for score and predictive sample options.
