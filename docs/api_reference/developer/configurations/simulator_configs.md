# Simulator Configurations

Simulator configurations control how continuous-time trajectories are solved.
Use `ODESimulatorConfig` for deterministic continuous-time dynamics and
`SDESimulatorConfig` for stochastic continuous-time dynamics. Discrete-time
simulation samples the transition distribution directly and does not accept a
simulator configuration.

## `SimulatorConfig`

::: dynestyx.configs.simulator.SimulatorConfig
    options:
      show_root_heading: false
      show_root_toc_entry: true

## `ODESimulatorConfig`

::: dynestyx.configs.simulator.ODESimulatorConfig
    options:
      show_root_heading: false
      show_root_toc_entry: true

## `SDESimulatorConfig`

::: dynestyx.configs.simulator.SDESimulatorConfig
    options:
      show_root_heading: false
      show_root_toc_entry: true
