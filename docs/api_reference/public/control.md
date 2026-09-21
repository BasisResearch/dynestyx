# Closed-loop control

Dynestyx can interleave simulation, observation, filtering, and control for a
single discrete-time trajectory. Which control an observation sees is set by
the model's `observation_control_alignment` -- the same field that governs
open-loop simulation. Closed-loop control currently implements
`"previous_transition"` only. Leaving the field unspecified (the default)
selects it with a warning; an explicit `"same_time"` raises
`NotImplementedError` -- we are working on it. Both conventions are described
below.

## `same_time` convention (not implemented yet)

Notation:

$$
\begin{aligned}
\tilde{p}_k &\approx p(x_k \mid y_0, \dots, y_{k-1},\ u_0, \dots, u_{k-1})
  && \text{is the predicted distribution,} \\
\hat{p}_k &\approx p(x_k \mid y_0, \dots, y_k,\ u_0, \dots, u_k)
  && \text{is the filtered distribution.}
\end{aligned}
$$

In this convention, the control $u_k$ drives the transition into the next state
$x_{k+1}$ and generates the observation $y_k$ (they are aligned).

$$
\begin{aligned}
& \text{Times} = [t_0, \dots, t_{N}]\\
&x_0 \sim p_0, \quad \tilde{p}_0 = p_0, \quad s_0 \text{ given} \quad \text{Initialization step} \\
&\text{for } k = 0,\dots N-2:\\
&\quad u_{k}, s_{k+1} =\pi(\tilde{p}_{k}, t_{k}, t_{k+1}, s_k)
  \quad \text{Select the control}\\
&\quad y_{k} \sim p(y_k \mid x_k, u_k, t_k) \quad \text{emit observation}\\
&\quad \hat{p}_k = \text{FilterAnalysis}(y_k, \tilde{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation} \\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
  \quad \text{State transition} \\
&\quad \tilde{p}_{k+1} = \text{PredictionUpdate}(\hat{p}_k, u_k)
  \quad \text{Predict the filtering distribution }\\
&k = N-1:\\
&\quad u_{k}, s_{k+1} =\pi(\tilde{p}_{k}, t_{k}, t_{k+1}, s_k)
  \quad \text{Select the control}\\
&\quad y_{k} \sim p(y_k \mid x_k, u_k, t_k) \quad \text{emit observation}\\
&\quad \hat{p}_k = \text{FilterAnalysis}(y_k, \tilde{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation}
\end{aligned}
$$

Here $s_0$ is `simulate`'s `initial_policy_state`: `None` by default, for a
stateless policy. It is never initialised for you, so a stateful policy must be
given its initial state explicitly -- for MPPI, `MPPI.initial_state()`.

Note the restrictions:

We have $N+1$ time steps $k = 0, 1, 2, \dots, N$.

The policy needs the last time $t_N$ to make its prediction at $t_{N-1}$, hence
we can have controls only up to time $t_{N-1}$.

Hence the last observation we obtain is at time $t_{N-1}$. At the last step, we
skip the state propagation (even though we could technically do it). This means
that we will have $N$ states, observations and controls, each of these aligned
on the grid $[t_0, \dots, t_{N-1}]$ (note the absence of the last time point).

This convention is similar to the open-loop convention in the sense that it
maintains the same number of states, controls and observations. It differs in
the sense that you specify an interval with $N+1$ elements but get $N$ elements
back. This is a fundamental limitation in requiring policies to know about the
next time step: it can never act at the final time step. The open loop does not
suffer from this because the controls are provided in advance, for each time
step.

An alternative is to run the loop:

$$
\begin{aligned}
& \text{Times} = [t_0, \dots, t_{N}]\\
&x_0 \sim p_0, \quad \tilde{p}_0 = p_0, \quad s_0 \text{ given} \quad \text{Initialization step} \\
&\text{for } k = 0,\dots N-1:\\
&\quad u_{k}, s_{k+1} =\pi(\tilde{p}_{k}, t_{k}, t_{k+1}, s_k)
  \quad \text{Select the control}\\
&\quad y_{k} \sim p(y_k \mid x_k, u_k, t_k) \quad \text{emit observation}\\
&\quad \hat{p}_k = \text{FilterAnalysis}(y_k, \tilde{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation} \\
&\quad x_{k+1} \sim p(x_{k+1} \mid x_k, u_k, t_k, t_{k+1})
  \quad \text{State transition} \\
&\quad \tilde{p}_{k+1} = \text{PredictionUpdate}(\hat{p}_k, u_k)
  \quad \text{Predict the filtering distribution }
\end{aligned}
$$

Here we obtain $N+1$ states but only $N$ observations and controls. The
advantage is that the states live on the provided time grid, but we break the
control/observations - state parity.

**This second loop is the one intended for closed-loop `"same_time"`, which
is not available yet** (it needs separate prediction and analysis filter
steps). Once it lands, states will span
$[t_0, \dots, t_N]$; observations, controls, filtered beliefs and policy states
will span $[t_0, \dots, t_{N-1}]$, reported by `ctrl_times`. Read the
alignment off that field rather than inferring it from array lengths: open-loop
`"previous_transition"` yields the same shapes with the observations at the
*other* end of the grid.

## `previous_transition` convention

Notation:

$$
\begin{aligned}
\tilde{p}_k &\approx p(x_k \mid y_1, \dots, y_{k-1},\ u_0, \dots, u_{k-1})
  && \text{is the predicted distribution,} \\
\hat{p}_k &\approx p(x_k \mid y_1, \dots, y_k,\ u_0, \dots, u_{k-1})
  && \text{is the filtered distribution.}
\end{aligned}
$$

In this convention, the control $u_k$ drives the transition into the next state
$x_{k+1}$ and generates the observation $y_{k+1}$. Hence, $y_0$ never exists.

$$
\begin{aligned}
&x_0 \sim p_0, \quad \tilde{p}_0 = \hat{p}_0 = p_0 \\
&u_k = \pi(\hat{p}_k) \\
&x_{k+1} \sim p(x_{k+1} \mid x_k, u_k) \\
&\tilde{p}_{k+1} = \text{PredictionUpdate}(\hat{p}_k, u_k) \\
&y_{k+1} \sim p(y_{k+1} \mid x_{k+1}, u_k) \\
&\hat{p}_{k+1} = \text{FilterUpdate}(y_{k+1}, \tilde{p}_{k+1}, u_k)
\end{aligned}
$$

As an algorithm on the grid, with the prediction and filter steps fused:

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
  \quad \text{emit observation}\\
&\quad \hat{p}_{k+1} = \text{FilterUpdate}(y_{k+1}, \hat{p}_k, u_k)
  \quad \text{Update the filtering distribution using the observation}
\end{aligned}
$$

Here the policy uses the filtering distribution. In this convention $y_0$ never
exists.

We have $N+1$ states (on the time grid $[t_0, \dots, t_{N}]$).

We have $N$ controls (on the time grid $[t_0, \dots, t_{N-1}]$); we can never
produce a control at time $t_N$.

We have $N$ observations (on the time grid $[t_1, \dots, t_{N}]$); we can never
produce an observation at time $t_0$, because $u_0$ is used for $x_1$ and $y_1$.

## Telling the two apart

Both conventions return $N+1$ states on $[t_0, \dots, t_N]$ and $N$ controls on
$[t_0, \dots, t_{N-1}]$. They differ in exactly one place:

| | `states` | `ctrl_times` | `obs_times` |
|---|---|---|---|
| `"same_time"` | $[t_0 \dots t_N]$ | $[t_0 \dots t_{N-1}]$ | $[t_0 \dots t_{N-1}]$ |
| `"previous_transition"` | $[t_0 \dots t_N]$ | $[t_0 \dots t_{N-1}]$ | $[t_1 \dots t_N]$ |

Array lengths are therefore identical and cannot identify which convention
produced a result. Read `obs_times` and `ctrl_times` off the result
rather than inferring alignment from shapes. `filtered_states_mean` also
differs: $N+1$ beliefs here, one per state, against $N$ under `"same_time"`.

## Current support

The first policy call acts on the model's initial-state distribution, which is
also $\tilde{p}_0$. A single-point grid gives the policy no later time to look
ahead to, so the loop body never runs: the result is one state and nothing
observed or controlled.

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
