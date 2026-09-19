# Layout

`Layout` maps a fixed pytree of array leaves to one flat trailing event axis.
It preserves shared leading axes, including time, simulations, and plates. Leaves
must have a common numeric dtype, and the example supplied to `from_example`
should contain event shapes only, without batch axes.

::: dynestyx.models.layout.Layout

Pass `state_layout`, `control_layout`, and `observation_layout` to
`DynamicalModel`. Each is independently optional and defaults to `None`.
Before calling a transition or observation operator, the model restores the
state and control structures. Without a corresponding layout, the input is
unchanged; `u=None` remains `None`.

Custom callables receive structured inputs but must return NumPyro distributions
over flat coordinates. `observation_layout` describes the observation output;
it does not change the state input to the observation function.

Gaussian and Dirac helpers accept structured function outputs and flatten them
using their output layouts. The model supplies omitted helper layouts from its
own declarations. When specifying structured covariance values at helper
construction, pass the output layout there too. Helpers do not unflatten their
inputs themselves, so direct helper calls must supply the expected structure.

Pass structured controls directly to `simulate`, `sample`, `condition`, or
`log_prob`. They are flattened internally before validation and backend processing:

```python
control_layout = dsx.Layout.from_example({"drive": jnp.zeros(2)})
# controls["drive"] has shape (n_times, 2)
ctrl_values = controls
```

A control layout determines `control_dim`; an explicit conflicting dimension is
rejected. Continuous simulation and MPPI do not yet support layouts.

`GaussianStateEvolution.cov` and `GaussianObservation.R` always mean variance:
a scalar gives independent noise of that variance at every coordinate; a matching
pytree gives per-coordinate variances. Full covariance matrices are accepted only
without an output layout, alongside scalar and vector variances. A matrix-shaped
leaf matching a layout still describes independent variances, not correlations.
Independent Gaussian observations use `GaussianObservation` with scalar or vector
`R`; no separate diagonal observation class is needed. Helpers accepting
`noise_std` convert those standard deviations to variances.
Constant structured noise is flattened at construction; callable transition noise
is evaluated on structured state and then flattened.

Initial distributions and observed data still use flat events. For example:

```python
layout = dsx.Layout.from_example(initial_state)
initial_condition = dsx.GaussianInitialCondition(initial_state, cov=0.01, state_layout=layout)
obs_values = observation_layout.flatten(structured_observations)
```

Simulation results return the declared pytrees by default in `states`, `x_0`,
`observations`, and corresponding `predicted_*` fields. Use `result.flatten().states` and `result.flatten().observations` when vector arrays are needed
(for example, `obs_values=result.flatten().observations[0]`). Without layouts, these
fields remain arrays. Both layouts are retained, and the earlier `structured_*`
names remain aliases. Filter/smoother results retain the layouts and expose
`structured_means`; latent-path results expose `structured_state_path`. All
leading axes are preserved; NumPyro sample-site values remain flat.

Simulators construct `SimulatedResult` from flat arrays with both layout keywords,
then call `.unflatten()`. This returns a structured copy; `.flatten()` returns a
flat copy. Both retain the layouts, leave times unchanged, and do nothing for
fields without a layout. Repeating either conversion is a no-op. Directly
constructed results are assumed to contain flat arrays until `.unflatten()` is
called.

Gaussian transitions and observations expand independent variances into dense
covariance matrices. Without a layout, a constant scalar variance is expanded
once the dimension is known during model construction. Callable process
covariances are expanded when evaluated. Gaussian initial conditions likewise construct a multivariate Normal with dense
covariance. Algorithm memory requirements
therefore remain unchanged: a layout does not make an EKF or UKF scalable to very
large states. Existing backend restrictions also remain: cd-dynamax discrete
EKF/UKF require constant process covariance and ignore absolute time arguments.
Use cuthbert for time-dependent discrete transitions. Exact Dirac models remain
subject to each inference algorithm's requirements on noise and transition density.

Custom transition and observation subclasses follow the same structured-input,
flat-distribution-output contract. Explicit linear-Gaussian matrix operators
continue to describe operations on vectors.
