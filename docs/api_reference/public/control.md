# Closed-loop control

Dynestyx can interleave simulation, observation, filtering, and control for a
single discrete-time trajectory. At each step it performs

\[
\begin{aligned}
x_0 &\sim p(x_0), \\
y_0 \mid x_0 &\sim p(y_0 \mid x_0, t_0), \\
\hat{x}_{0\mid0} &= \operatorname{FilterUpdate}(y_0, t_0), \\
(u_k, s_{k+1}) &= \pi(\hat{x}_{k\mid k}, t_k, t_{k+1}, s_k), \\
x_{k+1} \mid x_k,u_k &\sim p(x_{k+1}\mid x_k,u_k,t_k,t_{k+1}), \\
y_{k+1} \mid x_{k+1},u_k &\sim p(y_{k+1}\mid x_{k+1},u_k,t_{k+1}), \\
\hat{x}_{k+1\mid k+1} &= \operatorname{FilterUpdate}
  (\hat{x}_{k\mid k},u_k,y_{k+1},t_k,t_{k+1}).
\end{aligned}
\]

The observation at `t[k + 1]` receives `u[k]`, the control that produced its
state. This differs from the same-index convention used for a precomputed
open-loop control trajectory.

Controlled simulation currently supports one trajectory at a time. Its online
filter update is implemented with Cuthbert and supports `KFConfig`, `EKFConfig`,
`EnKFConfig`, and `PFConfig`. `dsx.plate` and `n_simulations > 1` are rejected
explicitly. Controlled simulation requires `filter_source="cuthbert"` and
rejects configurations that request another backend.

## Simulator and policy protocol

::: dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.PolicyCallable
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.ControlledSimulatedResult
    options:
      show_root_heading: true

## Policy helpers

::: dynestyx.control.discrete_controller_simulators.filter_state_mean
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.filter_state_dist
    options:
      show_root_heading: true

## MPPI-inspired policy

::: dynestyx.control.mppi.MPPI
    options:
      show_root_heading: true
