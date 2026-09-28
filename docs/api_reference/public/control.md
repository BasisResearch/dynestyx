# Closed-loop control

Dynestyx can combine simulation, observation, state estimation, and closed-loop (or online) control for a
single discrete-time trajectory. Which control an observation sees is set by
the model's `observation_control_alignment` (the same field that governs
open-loop simulation). Unlike open-loop control, closed-loop control defaults to `"previous_transition"`
and raises a warning if the alignment is not specified. `"same_time"` with state estimation is currently not supported; its support is tracked in [Issue #372](https://github.com/BasisResearch/dynestyx/issues/372).

Both conventions (with and without state estimation) are described below.

In the following, $s_0$ is `simulate`'s `initial_policy_state`: `None` by default, for a
stateless policy. It must be initialized and passed explicitly if the policy requires it. By default, the loop runs with state estimation; pass `use_true_state=True` to run it on the true state instead.


## `previous_transition` convention (default for closed-loop control)

In this convention, the control $u_k$ drives the transition into the next state
$x_{k+1}$ and generates the observation $y_{k+1}$. Hence, $y_0$ never exists.

### With the true state

On $\text{Times} = [t_0, \dots, t_N]$:

$$
\begin{aligned}
&x_0 \sim p_0, \quad s_0 \text{ given}
    && \text{Initialization step} \\
&\text{for } k = 0, \dots, N-1: \\
&\quad u_k, s_{k+1} = \pi(x_k, t_k, t_{k+1}, s_k)
    && \text{Select the control} \\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
    && \text{State transition} \\
&\quad y_{k+1} \sim p(y_{k+1} \mid x_{k+1}, u_k, t_{k+1})
    && \text{Emit observation}
\end{aligned}
$$

At each step, the true state $x_k$ is passed to the policy as a
[`Delta`](https://num.pyro.ai/en/stable/distributions.html#delta) distribution (its `.mean` is the state itself).


### With an estimated state

Notation:

$$
\begin{aligned}
\tilde{p}_k &\approx p(x_k \mid y_1, \dots, y_{k-1},\ u_0, \dots, u_{k-1})
  && \text{is the predicted distribution,} \\
\hat{p}_k &\approx p(x_k \mid y_1, \dots, y_k,\ u_0, \dots, u_{k-1})
  && \text{is the filtered distribution.}
\end{aligned}
$$

The loop is:

$$
\begin{aligned}
& \text{Times} = [t_0, \dots, t_{N}]\\
&x_0 \sim p_0, \quad \hat{p}_0 = p_0, \quad s_0 \text{ given} \quad \text{Initialization step} \\
&\text{for } k = 0,\dots N-1:\\
&\quad u_{k}, s_{k+1} =\pi(\hat{p}_{k}, t_{k}, t_{k+1}, s_k)
  \quad \text{Select the control}\\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
  \quad \text{State transition} \\
&\quad y_{k+1} \sim p(y_{k+1} \mid x_{k+1}, u_k, t_{k+1})
  \quad \text{Emit observation}\\
&\quad \hat{p}_{k+1} = \text{FilterUpdate}(y_{k+1}, \hat{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation}
\end{aligned}
$$

At each step, an estimate of the state, the filtered distribution $\hat{p}_k$, is passed to the policy as a `Distribution`.

In the `previous_transition` convention:

  1. There are $N+1$ states $x_0, x_1, \dots, x_{N}$ on the time grid $[t_0, \dots, t_N]$ (reported in the results as `times`).
  2. There are $N$ controls $u_0, u_1, \dots, u_{N-1}$ on the time grid $[t_0, \dots, t_{N-1}]$ (reported in the results as `ctrl_times`).
  3. There are $N$ observations $y_1, \dots, y_{N}$ on the time grid $[t_1, \dots, t_{N}]$ (reported in the results as `obs_times`).



## `same_time` convention

In this convention, the control $u_k$ drives the transition into the next state
$x_{k+1}$ and generates the observation $y_k$. Hence, the last observation $y_{N}$ never exists.

### With the true state

On $\text{Times} = [t_0, \dots, t_N]$:

$$
\begin{aligned}
&x_0 \sim p_0, \quad s_0 \text{ given}
    && \text{Initialization step} \\
&\text{for } k = 0, \dots, N-1: \\
&\quad u_k, s_{k+1} = \pi(x_k, t_k, t_{k+1}, s_k)
    && \text{Select the control} \\
&\quad y_k \sim p(y_k \mid x_k, u_k, t_k)
    && \text{Emit observation} \\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
    && \text{State transition}
\end{aligned}
$$

At each step, the true state $x_k$ is passed to the policy as a
[`Delta`](https://num.pyro.ai/en/stable/distributions.html#delta) distribution (its `.mean` is the state itself).


### With an estimated state (not implemented)

Notation:

$$
\begin{aligned}
\tilde{p}_k &\approx p(x_k \mid y_0, \dots, y_{k-1},\ u_0, \dots, u_{k-1})
  && \text{is the predicted distribution,} \\
\hat{p}_k &\approx p(x_k \mid y_0, \dots, y_k,\ u_0, \dots, u_k)
  && \text{is the filtered distribution.}
\end{aligned}
$$

The loop is:

$$
\begin{aligned}
& \text{Times} = [t_0, \dots, t_{N}]\\
&x_0 \sim p_0, \quad \tilde{p}_0 = p_0, \quad s_0 \text{ given} \quad \text{Initialization step} \\
&\text{for } k = 0,\dots N-1:\\
&\quad u_{k}, s_{k+1} =\pi(\tilde{p}_{k}, t_{k}, t_{k+1}, s_k)
  \quad \text{Select the control}\\
&\quad y_{k} \sim p(y_k \mid x_k, u_k, t_k) \quad \text{Emit observation}\\
&\quad \hat{p}_k = \text{FilterAnalysis}(y_k, \tilde{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation} \\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
  \quad \text{State transition} \\
&\quad \tilde{p}_{k+1} = \text{PredictionUpdate}(\hat{p}_k, u_k)
  \quad \text{Predict the filtering distribution}
\end{aligned}
$$

At each step, an estimate of the state, the predicted distribution $\tilde{p}_k$, is passed to the policy as a `Distribution`.


In the `same_time` convention:

  1. There are $N+1$ states $x_0, x_1, \dots, x_{N}$ on the time grid $[t_0, \dots, t_N]$ (reported in the results as `times`).
  2. There are $N$ controls $u_0, u_1, \dots, u_{N-1}$ on the time grid $[t_0, \dots, t_{N-1}]$ (reported in the results as `ctrl_times`).
  3. There are $N$ observations $y_0, y_1, \dots, y_{N-1}$ on the time grid $[t_0, \dots, t_{N-1}]$ (reported in the results as `obs_times`).



## Telling the two apart

Both conventions return $N+1$ states on $[t_0, \dots, t_N]$ and $N$ controls on
$[t_0, \dots, t_{N-1}]$. They differ in exactly one place:

| | `states` | `ctrl_times` | `obs_times` |
|---|---|---|---|
| `"same_time"` | $[t_0 \dots t_N]$ | $[t_0 \dots t_{N-1}]$ | $[t_0 \dots t_{N-1}]$ |
| `"previous_transition"` | $[t_0 \dots t_N]$ | $[t_0 \dots t_{N-1}]$ | $[t_1 \dots t_N]$ |

Array lengths are therefore identical and cannot identify which convention
produced a result. Read `obs_times` and `ctrl_times` off the result
rather than inferring alignment from shapes. With state estimation,
`filtered_states_mean` also differs: $N+1$ beliefs under `"previous_transition"`
(one per state), against $N$ under `"same_time"`. On the true state, it is `None`.


## Simulator and policy protocol

::: dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.ControlledSimulatedResult
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.PolicyCallable
    options:
      show_root_heading: true


## MPPI policy

::: dynestyx.control.mppi.MPPI
    options:
      show_root_heading: true

## Policy helpers

::: dynestyx.control.discrete_controller_simulators.filter_state_mean
    options:
      show_root_heading: true

::: dynestyx.control.discrete_controller_simulators.filter_state_dist
    options:
      show_root_heading: true
