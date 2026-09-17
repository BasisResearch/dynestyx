# State layour

`Layout` maps a fixed pytree of array leaves to one flat trailing event axis.
It preserves shared leading axes, including time, simulations, and plates. Leaves
must have a common numeric dtype, and the example supplied to `from_example`
should contain event shapes only, without batch axes.

::: dynestyx.models.layout.Layout

Pass `state_layout=layout` to `GaussianStateEvolution` or `DiracStateEvolution`.
Their `F` function receives and returns structured states. Observation operators
receive structured states through `state_layout`, and can return a different
structure through `observation_layout`. Without an observation layout, they
continue returning scalar or flat observations. Controls remain flat.

Also pass `state_layout=layout` and `observation_layout=observation_layout` to
`DynamicalModel` to declare the result structures. These model fields default to
`None`; they are not inferred from the component layouts. This also supports
plain callable transitions and observations, which still operate on flat events.

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
initial_condition = dist.Normal(layout.flatten(initial_state), 0.1).to_event(1)
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

Independent Gaussian distributions are used internally. Filter backends that need
dense covariance matrices receive diagonal matrices. Algorithm memory requirements
therefore remain unchanged: a layout does not make an EKF or UKF scalable to very
large states. Existing backend restrictions also remain: cd-dynamax discrete
EKF/UKF require constant process covariance and ignore absolute time arguments.
Use cuthbert for time-dependent discrete transitions. Exact Dirac models remain
subject to each inference algorithm's requirements on noise and transition density.

Existing custom subclasses keep their `__call__` contracts and can opt into the
protected layout helpers. `LinearGaussianStateEvolution`,
`LinearGaussianObservation`, and `FieldLayout` retain their existing APIs.
