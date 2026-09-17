# Closed-loop control

Dynestyx can interleave simulation, observation, filtering, and control for a
single discrete-time trajectory. At each step it performs

$$
\begin{aligned}
x_0 &\sim p(x_0), \\
\hat p_0 &= p(x_0), \\
(u_k, s_{k+1}) &= \pi(\hat p_k, t_k, t_{k+1}, s_k), \\
x_{k+1} \mid x_k,u_k &\sim p(x_{k+1}\mid x_k,u_k,t_k,t_{k+1}), \\
y_{k+1} \mid x_{k+1},u_k &\sim p(y_{k+1}\mid x_{k+1},u_k,t_{k+1}), \\
\hat p_{k+1} &= \operatorname{FilterUpdate}
  (\hat p_k,u_k,y_{k+1},t_k,t_{k+1}).
\end{aligned}
$$

The initial policy decision uses the model's initial-state distribution as its
belief; no synthetic initial observation is generated. Every observation at
`t[k + 1]` receives `u[k]`, the control that produced its state. Closed-loop
simulation therefore always follows the `"previous_transition"` convention,
independently of `dynamics.observation_control_alignment`. For `T` prediction
times, the result contains `T` states and filtered beliefs but `T - 1`
observations and controls.

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
