# MPPI

::: dynestyx.control.mppi.MPPI
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source

## MPPIState

::: dynestyx.control.mppi.MPPIState
    options:
      show_root_heading: false
      show_root_toc_entry: false

## MPPIStepInfo

::: dynestyx.control.mppi.MPPIStepInfo
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Noise distributions

NumPyro distributions for the perturbations `MPPI` adds to its nominal control
sequence. Pass one as `MPPI(noise=...)`, or any other NumPyro distribution
whose samples have shape `(horizon_length, control_dim)`.

### WhiteNoise

::: dynestyx.control.utils.distribution_utils.WhiteNoise
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

### AR1Noise

::: dynestyx.control.utils.distribution_utils.AR1Noise
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

### ColoredNoise

::: dynestyx.control.utils.distribution_utils.ColoredNoise
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
