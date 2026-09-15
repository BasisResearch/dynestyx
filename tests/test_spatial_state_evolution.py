"""Tests for `dynestyx.models.spatial`."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import dynestyx as dsx
from dynestyx.inference.configs.filter import EKFConfig, EnKFConfig
from dynestyx.models.spatial import (
    FieldLayout,
    SpatialStateEvolution,
    field_observation,
    spatial_dynamics,
)

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- layout


def test_layout_single_field_shape_and_dim():
    layout = FieldLayout((3, 4, 5, 6))
    assert layout.shapes == ((3, 4, 5, 6),)
    assert layout.state_dim == 3 * 4 * 5 * 6
    assert layout.is_single
    assert layout.n_fields == 1


def test_layout_multi_field_offsets():
    layout = FieldLayout(((2, 8, 8), (1, 16, 16)))
    assert layout.shapes == ((2, 8, 8), (1, 16, 16))
    assert layout.sizes == (128, 256)
    assert layout.offsets == (0, 128)
    assert layout.state_dim == 384
    assert not layout.is_single


def test_layout_roundtrip_single():
    layout = FieldLayout((3, 4, 5))
    field = jr.normal(jr.PRNGKey(0), (3, 4, 5))
    flat = layout.flatten(field)
    assert flat.shape == (60,)
    out = layout.unflatten(flat)
    assert isinstance(out, jax.Array)
    assert jnp.array_equal(out, field)


def test_layout_roundtrip_multi():
    layout = FieldLayout(((2, 8, 8), (1, 16, 16)))
    fields = (
        jr.normal(jr.PRNGKey(0), (2, 8, 8)),
        jr.normal(jr.PRNGKey(1), (1, 16, 16)),
    )
    out = layout.unflatten(layout.flatten(fields))
    assert isinstance(out, tuple) and len(out) == 2
    for a, b in zip(out, fields, strict=True):
        assert jnp.array_equal(a, b)


@pytest.mark.parametrize("batch", [(), (7,), (3, 7)])
def test_layout_unflatten_preserves_batch_axes(batch):
    layout = FieldLayout(((2, 4, 4), (1, 4, 4)))
    x = jr.normal(jr.PRNGKey(0), (*batch, layout.state_dim))
    a, b = layout.unflatten(x)
    assert a.shape == (*batch, 2, 4, 4)
    assert b.shape == (*batch, 1, 4, 4)
    assert jnp.allclose(layout.flatten((a, b)), x)


def test_layout_matches_hand_written_reshape():
    """The single-field case must agree with the notebooks' manual reshape."""
    layout = FieldLayout((3, 8, 8, 8))
    x = jr.normal(jr.PRNGKey(0), (layout.state_dim,))
    out = layout.unflatten(x)
    assert isinstance(out, jax.Array)
    assert jnp.array_equal(out, x.reshape(3, 8, 8, 8))


def test_layout_rejects_bad_shapes():
    with pytest.raises(ValueError, match="non-empty"):
        FieldLayout(())
    with pytest.raises(ValueError, match="strictly positive"):
        FieldLayout((3, 0, 4))
    layout = FieldLayout((2, 3))
    with pytest.raises(ValueError, match="trailing axis"):
        layout.unflatten(jnp.zeros(5))
    with pytest.raises(ValueError, match="trailing shape"):
        layout.flatten(jnp.zeros((4, 4)))


def test_layout_rejects_inconsistent_batch_axes():
    layout = FieldLayout(((1, 4), (1, 4)))
    with pytest.raises(ValueError, match="Inconsistent batch axes"):
        layout.flatten((jnp.zeros((3, 1, 4)), jnp.zeros((2, 1, 4))))


# --------------------------------------------------------------- transition mechanics


def _decay_stepper(factor=0.9):
    return lambda fields: factor * fields


def test_substeps_equal_manual_applications():
    layout_shape = (2, 4, 4)
    stepper = _decay_stepper()
    evo = SpatialStateEvolution(
        stepper, field_shape=layout_shape, dt_obs=1.0, n_substeps=5
    )
    x = jr.normal(jr.PRNGKey(0), (evo.state_dim,))

    manual = evo.layout.unflatten(x)
    for _ in range(5):
        manual = stepper(manual)

    out = evo(x=x, u=None, t_now=0.0, t_next=1.0).mean
    assert jnp.array_equal(out, evo.layout.flatten(manual))


def test_single_substep_matches_scan_path():
    """n_substeps=1 takes a direct call; it must agree with the scan for n=1."""
    stepper = _decay_stepper(0.75)
    x = jr.normal(jr.PRNGKey(0), (32,))
    direct = SpatialStateEvolution(
        stepper, field_shape=(2, 4, 4), dt_obs=1.0, n_substeps=1
    )
    out_direct = direct(x=x, u=None, t_now=0.0, t_next=1.0).mean
    expected = direct.layout.flatten(stepper(direct.layout.unflatten(x)))
    assert jnp.array_equal(out_direct, expected)


def test_multi_field_stepper():
    layout = FieldLayout(((2, 8, 8), (1, 16, 16)))

    def stepper(fields):
        a, b = fields
        return (0.5 * a, b + 1.0)

    evo = SpatialStateEvolution(
        stepper, field_shape=((2, 8, 8), (1, 16, 16)), dt_obs=0.5, n_substeps=3
    )
    x = jnp.zeros(layout.state_dim)
    a, b = evo.layout.unflatten(evo(x=x, u=None, t_now=0.0, t_next=0.5).mean)
    assert jnp.allclose(a, 0.0)
    assert jnp.allclose(b, 3.0)


def test_dt_obs_mismatch_raises():
    evo = SpatialStateEvolution(
        _decay_stepper(), field_shape=(1, 4), dt_obs=0.5, n_substeps=1
    )
    x = jnp.zeros(evo.state_dim)
    evo(x=x, u=None, t_now=0.0, t_next=0.5)  # matching interval is fine
    with pytest.raises(ValueError, match="dt_obs"):
        evo(x=x, u=None, t_now=0.0, t_next=0.25)


def test_invalid_construction():
    with pytest.raises(ValueError, match="n_substeps"):
        SpatialStateEvolution(
            _decay_stepper(), field_shape=(1, 4), dt_obs=1.0, n_substeps=0
        )
    with pytest.raises(ValueError, match="dt_obs"):
        SpatialStateEvolution(_decay_stepper(), field_shape=(1, 4), dt_obs=0.0)


def test_construction_validates_stepper_output_shape():
    with pytest.raises(ValueError, match="field_shape declares"):
        SpatialStateEvolution(
            lambda fields: jnp.zeros((1, 5)), field_shape=(1, 4), dt_obs=1.0
        )
    with pytest.raises(ValueError, match="field_shape declares"):
        SpatialStateEvolution(
            lambda fields: (fields, fields), field_shape=(1, 4), dt_obs=1.0
        )
    # opt out for a stepper that cannot be abstractly traced
    SpatialStateEvolution(
        lambda fields: jnp.zeros((1, 5)),
        field_shape=(1, 4),
        dt_obs=1.0,
        validate=False,
    )


def test_construction_runs_no_solver_work():
    """Validation is abstract: building the model must not execute the stepper."""
    calls = {"n": 0}

    def counting_stepper(fields):
        calls["n"] += 1
        return fields

    evo = SpatialStateEvolution(
        counting_stepper, field_shape=(1, 4), dt_obs=0.5, n_substeps=8
    )
    dsx.DynamicalModel(
        initial_condition=dist.Delta(jnp.zeros(4), event_dim=1),
        state_evolution=evo,
        observation_model=dsx.DiracIdentityObservation(),
        control_dim=0,
    )
    # eval_shape traces the stepper once abstractly; it must never run it 8 times,
    # and DynamicalModel must not re-run it at all.
    assert calls["n"] <= 1


def test_dt_obs_other_than_one_is_constructible():
    """DynamicalModel's probe uses a unit interval; dt_obs != 1.0 must still work."""
    dynamics = spatial_dynamics(
        _decay_stepper(),
        field_shape=(1, 4),
        dt_obs=0.25,
        initial_field=jnp.ones((1, 4)),
    )
    assert dynamics.state_dim == 4


def test_stepper_is_a_dynamic_pytree_leaf():
    """Array-valued solver parameters must stay visible to JAX, not be baked in."""

    class ArrayStepper(eqx.Module):
        gain: jax.Array

        def __call__(self, fields):
            return self.gain * fields

    evo = SpatialStateEvolution(
        ArrayStepper(jnp.array(0.5)), field_shape=(1, 4), dt_obs=1.0
    )
    leaves = [leaf for leaf in jax.tree.leaves(evo) if eqx.is_array(leaf)]
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], jnp.array(0.5))


# ------------------------------------------------------------------ auxiliary carry


def test_aux_advances_within_a_transition_and_resets_between():
    """aux is a scan carry inside one transition, rebuilt at every boundary."""

    def stepper(fields, aux):
        return fields + aux, aux + 1.0

    evo = SpatialStateEvolution(
        stepper,
        field_shape=(1, 3),
        dt_obs=1.0,
        n_substeps=4,
        aux_init=lambda fields: jnp.array(1.0),
    )
    x = jnp.zeros(3)

    # within one transition aux counts 1, 2, 3, 4 -> field gains 1+2+3+4 = 10
    first = evo(x=x, u=None, t_now=0.0, t_next=1.0).mean
    assert jnp.allclose(first, 10.0)

    # the next transition restarts aux at 1, so it gains 10 again (not 5+6+7+8)
    second = evo(x=first, u=None, t_now=1.0, t_next=2.0).mean
    assert jnp.allclose(second, 20.0)

    _, final_aux = evo.rollout(evo.layout.unflatten(x))
    assert jnp.allclose(final_aux, 5.0)


def test_aux_with_single_substep():
    def stepper(fields, aux):
        return fields + aux, aux + 1.0

    evo = SpatialStateEvolution(
        stepper,
        field_shape=(1, 2),
        dt_obs=1.0,
        n_substeps=1,
        aux_init=lambda fields: jnp.array(2.0),
    )
    out = evo(x=jnp.zeros(2), u=None, t_now=0.0, t_next=1.0).mean
    assert jnp.allclose(out, 2.0)


# ----------------------------------------------------------------------- end-to-end


def test_spatial_dynamics_simulate_matches_hand_rolled_rollout():
    field_shape = (2, 4, 4)
    stepper = _decay_stepper(0.8)
    initial_field = jr.normal(jr.PRNGKey(0), field_shape)

    dynamics = spatial_dynamics(
        stepper,
        field_shape=field_shape,
        dt_obs=0.5,
        initial_field=initial_field,
        n_substeps=2,
    )
    assert dynamics.state_dim == 32
    assert dynamics.observation_dim == 32

    n_steps = 6
    times = jnp.arange(n_steps, dtype=jnp.float64) * 0.5
    sim = dsx.simulate(dynamics, rng_key=jr.PRNGKey(1), predict_times=times)
    assert sim.states is not None and sim.observations is not None
    states = sim.states[0]
    assert states.shape == (n_steps, 32)
    assert bool(jnp.isfinite(states).all())

    expected = initial_field
    for k in range(n_steps):
        assert jnp.allclose(states[k], expected.reshape(-1))
        expected = stepper(stepper(expected))

    # identity observation: y == x
    assert jnp.allclose(sim.observations[0], states)


def test_simulate_multi_field():
    def stepper(fields):
        a, b = fields
        return (0.9 * a, 0.5 * b)

    initial = (jnp.ones((2, 4, 4)), jnp.ones((1, 8, 8)))
    dynamics = spatial_dynamics(
        stepper,
        field_shape=((2, 4, 4), (1, 8, 8)),
        dt_obs=1.0,
        initial_field=initial,
        n_substeps=1,
    )
    sim = dsx.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=jnp.arange(3.0))
    assert sim.states is not None
    layout = FieldLayout(((2, 4, 4), (1, 8, 8)))
    a, b = layout.unflatten(sim.states[0])  # batched unflatten over time
    assert a.shape == (3, 2, 4, 4)
    assert b.shape == (3, 1, 8, 8)
    assert jnp.allclose(a[2], 0.9**2)
    assert jnp.allclose(b[2], 0.5**2)


def test_external_non_jax_stepper_via_pure_callback():
    """An external solver reached through jax.pure_callback works under the simulator."""
    calls = {"n": 0}

    def numpy_step(field):
        calls["n"] += 1
        arr = np.asarray(field)  # JAX hands over an ArrayImpl, not an ndarray
        return 0.5 * arr

    shape_dtype = jax.ShapeDtypeStruct((1, 4), jnp.float64)

    def stepper(fields):
        # vmap_method is required: the simulator vmaps over n_simulations even when it is 1.
        return jax.pure_callback(
            numpy_step, shape_dtype, fields, vmap_method="sequential"
        )

    dynamics = spatial_dynamics(
        stepper,
        field_shape=(1, 4),
        dt_obs=1.0,
        initial_field=jnp.ones((1, 4)),
        n_substeps=2,
    )
    sim = dsx.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=jnp.arange(3.0))
    assert calls["n"] > 0
    assert sim.states is not None
    assert jnp.allclose(sim.states[0][2], 0.5**4)


# ------------------------------------------------------------------------ noise


def test_process_noise_none_is_deterministic():
    evo = SpatialStateEvolution(_decay_stepper(), field_shape=(1, 4), dt_obs=1.0)
    assert evo.process_noise_scale is None
    assert isinstance(evo(x=jnp.zeros(4), u=None, t_now=0.0, t_next=1.0), dist.Delta)


def test_process_noise_scalar_stays_scalar_and_centers_on_the_stepped_field():
    """A scalar std must not materialise a (state_dim,) vector."""
    stepper = _decay_stepper(0.8)
    evo = SpatialStateEvolution(
        stepper, field_shape=(2, 4, 4), dt_obs=1.0, process_noise_std=0.1
    )
    scale = evo.process_noise_scale
    assert scale is not None and scale.shape == ()

    x = jr.normal(jr.PRNGKey(0), (evo.state_dim,))
    d = evo(x=x, u=None, t_now=0.0, t_next=1.0)
    assert d.event_shape == (evo.state_dim,)
    assert d.batch_shape == ()
    assert jnp.allclose(d.mean, evo.layout.flatten(stepper(evo.layout.unflatten(x))))

    samples = d.sample(jr.PRNGKey(1), (4000,))
    assert jnp.allclose(samples.std(axis=0).mean(), 0.1, atol=5e-3)


def test_process_noise_per_field_broadcasts():
    def stepper(fields):
        a, b = fields
        return (0.9 * a, 0.5 * b)

    evo = SpatialStateEvolution(
        stepper,
        field_shape=((2, 4, 4), (1, 8, 8)),
        dt_obs=1.0,
        process_noise_std=(0.1, 0.3),
    )
    scale = evo.process_noise_scale
    assert scale is not None
    assert scale.shape == (evo.state_dim,)
    assert jnp.allclose(scale[:32], 0.1)
    assert jnp.allclose(scale[32:], 0.3)


def test_process_noise_per_component_array():
    """A per-component std broadcasts across the spatial axes of its field."""
    std = jnp.array([0.1, 0.2, 0.3])[:, None, None]
    evo = SpatialStateEvolution(
        _decay_stepper(), field_shape=(3, 4, 4), dt_obs=1.0, process_noise_std=std
    )
    scale = evo.process_noise_scale
    assert scale is not None
    per_component = evo.layout.unflatten(scale)
    assert isinstance(per_component, jax.Array)
    assert per_component.shape == (3, 4, 4)
    for i, expected in enumerate((0.1, 0.2, 0.3)):
        assert jnp.allclose(per_component[i], expected)


def test_process_noise_rejects_bad_specs():
    with pytest.raises(ValueError, match="strictly positive"):
        SpatialStateEvolution(
            _decay_stepper(), field_shape=(1, 4), dt_obs=1.0, process_noise_std=0.0
        )
    with pytest.raises(ValueError, match="one entry per field"):
        SpatialStateEvolution(
            _decay_stepper(),
            field_shape=(1, 4),
            dt_obs=1.0,
            process_noise_std=(0.1, 0.2),
        )
    with pytest.raises(ValueError, match="does not broadcast"):
        SpatialStateEvolution(
            _decay_stepper(),
            field_shape=(1, 4),
            dt_obs=1.0,
            process_noise_std=jnp.ones(3),
        )
    with pytest.raises(ValueError, match="multi-field layout"):
        SpatialStateEvolution(
            lambda fields: fields,
            field_shape=((1, 4), (1, 4)),
            dt_obs=1.0,
            process_noise_std=jnp.ones(8),
        )


def test_noise_scale_is_a_dynamic_pytree_leaf():
    """Noise scales are arrays, so they must stay reachable as pytree leaves."""
    evo = SpatialStateEvolution(
        _decay_stepper(), field_shape=(2, 4), dt_obs=1.0, process_noise_std=(0.1,) * 1
    )
    leaves = jax.tree.leaves(eqx.filter(evo, eqx.is_array))
    assert any(leaf.shape == (evo.state_dim,) for leaf in leaves)


def test_field_observation_switches_between_dirac_and_gaussian():
    dirac = field_observation(field_shape=(1, 4))
    assert isinstance(dirac, dsx.DiracIdentityObservation)
    assert isinstance(dirac(jnp.zeros(4), None, 0.0), dist.Delta)

    gauss = field_observation(field_shape=(1, 4), noise_std=0.25)
    assert isinstance(gauss, dsx.DiagonalGaussianObservation)
    d = gauss(jnp.zeros(4), None, 0.0)
    assert d.event_shape == (4,)
    assert jnp.allclose(d.mean, 0.0)
    assert jnp.allclose(d.variance, 0.25**2)


def test_field_observation_per_field_noise():
    obs = field_observation(field_shape=((1, 4), (1, 4)), noise_std=(0.1, 0.5))
    d = obs(jnp.zeros(8), None, 0.0)
    assert jnp.allclose(d.variance[:4], 0.1**2)
    assert jnp.allclose(d.variance[4:], 0.5**2)


def test_diagonal_gaussian_observation_with_a_measurement_function():
    """A nonlinear/subsampling operator needs no dense (d_y, d_y) covariance."""
    obs = dsx.DiagonalGaussianObservation(0.1, h=lambda x, u, t: x[:3])
    d = obs(jnp.arange(10.0), None, 0.0)
    assert d.event_shape == (3,)
    assert jnp.allclose(d.mean, jnp.arange(3.0))


def test_simulate_recovers_both_noise_levels():
    """Simulated spread matches the requested process and observation stds."""
    field_shape = (1, 4, 4)
    decay = 0.9
    times = jnp.arange(4, dtype=jnp.float64)
    dynamics = spatial_dynamics(
        _decay_stepper(decay),
        field_shape=field_shape,
        dt_obs=1.0,
        initial_field=jnp.ones(field_shape),
        process_noise_std=0.05,
        observation_noise_std=0.2,
    )
    sim = dsx.simulate(
        dynamics, rng_key=jr.PRNGKey(0), predict_times=times, n_simulations=2000
    )
    assert sim.states is not None and sim.observations is not None

    # states[0] is the (deterministic) initial condition, so the first transition
    # is states[:, 1] - decay * states[:, 0].
    increments = sim.states[:, 1] - decay * sim.states[:, 0]
    assert jnp.allclose(increments.mean(), 0.0, atol=5e-3)
    assert jnp.allclose(increments.std(), 0.05, atol=5e-3)

    residuals = sim.observations - sim.states
    assert jnp.allclose(residuals.mean(), 0.0, atol=1e-2)
    assert jnp.allclose(residuals.std(), 0.2, atol=1e-2)


def test_noise_makes_log_prob_non_degenerate():
    """The point of Gaussian noise: `dsx.log_prob` stops being a degenerate 0/-inf."""
    field_shape = (1, 4)
    times = jnp.arange(4, dtype=jnp.float64)
    noisy = spatial_dynamics(
        _decay_stepper(),
        field_shape=field_shape,
        dt_obs=1.0,
        initial_field=jnp.ones(field_shape),
        process_noise_std=0.05,
        observation_noise_std=0.2,
    )
    sim = dsx.simulate(noisy, rng_key=jr.PRNGKey(0), predict_times=times)
    assert sim.states is not None and sim.observations is not None

    lp = dsx.log_prob(
        noisy,
        state_path_params=sim.states[0],
        state_path_param_times=times,
        obs_times=times,
        obs_values=sim.observations[0],
    )
    assert bool(jnp.isfinite(lp)) and float(lp) != 0.0

    # A state path that does not follow the dynamics is merely less likely, where the
    # deterministic model would give -inf. The initial condition stays a Delta, so only
    # the transitions are perturbed.
    perturbed = sim.states[0].at[1:].add(0.01)
    lp_perturbed = dsx.log_prob(
        noisy,
        state_path_params=perturbed,
        state_path_param_times=times,
        obs_times=times,
        obs_values=sim.observations[0],
    )
    assert bool(jnp.isfinite(lp_perturbed))
    assert float(lp_perturbed) < float(lp)


def test_enkf_runs_on_a_noisy_spatial_model():
    field_shape = (1, 4, 4)
    times = jnp.arange(6, dtype=jnp.float64)
    dynamics = spatial_dynamics(
        _decay_stepper(0.95),
        field_shape=field_shape,
        dt_obs=1.0,
        initial_field=jnp.ones(field_shape),
        process_noise_std=0.05,
        observation_noise_std=0.2,
    )
    sim = dsx.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=times)
    assert sim.observations is not None

    with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=jr.PRNGKey(1)):
        with dsx.Filter(filter_config=EnKFConfig(n_particles=32)):
            dsx.sample("f", dynamics, obs_times=times, obs_values=sim.observations[0])

    mean = tr["f_filtered_states_mean"]["value"]
    assert mean.shape == (len(times), dynamics.state_dim)
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(tr["f_marginal_loglik"]["value"]))


def test_noisy_subsampling_observation_of_a_field():
    """A probe-point operator plus diagonal noise: observation_dim < state_dim."""
    field_shape = (2, 8, 8)
    layout = FieldLayout(field_shape)
    probe_index = jnp.array([0, 5, 63, 100])

    evolution = SpatialStateEvolution(
        _decay_stepper(0.9),
        field_shape=field_shape,
        dt_obs=1.0,
        process_noise_std=0.05,
    )
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.Delta(
            layout.flatten(jnp.ones(field_shape)), event_dim=1
        ),
        state_evolution=evolution,
        observation_model=dsx.DiagonalGaussianObservation(
            0.1, h=lambda x, u, t: x[probe_index]
        ),
        control_dim=0,
    )
    assert dynamics.state_dim == 128
    assert dynamics.observation_dim == 4

    sim = dsx.simulate(
        dynamics,
        rng_key=jr.PRNGKey(0),
        predict_times=jnp.arange(3, dtype=jnp.float64),
        n_simulations=500,
    )
    assert sim.states is not None and sim.observations is not None
    assert sim.observations.shape == (500, 3, 4)
    residuals = sim.observations - sim.states[..., probe_index]
    assert jnp.allclose(residuals.std(), 0.1, atol=1e-2)


def test_initial_std_switches_the_initial_condition():
    field_shape = (1, 4)
    times = jnp.arange(3, dtype=jnp.float64)
    exact = spatial_dynamics(
        _decay_stepper(),
        field_shape=field_shape,
        dt_obs=1.0,
        initial_field=jnp.ones(field_shape),
    )
    assert isinstance(exact.initial_condition, dist.Delta)

    spread = spatial_dynamics(
        _decay_stepper(),
        field_shape=field_shape,
        dt_obs=1.0,
        initial_field=jnp.ones(field_shape),
        initial_std=0.1,
    )
    assert spread.state_dim == 4
    sim = dsx.simulate(
        spread, rng_key=jr.PRNGKey(0), predict_times=times, n_simulations=2000
    )
    assert sim.states is not None
    assert jnp.allclose(sim.states[:, 0].mean(), 1.0, atol=1e-2)
    assert jnp.allclose(sim.states[:, 0].std(), 0.1, atol=1e-2)


def test_ekf_runs_when_every_component_is_gaussian():
    """A Delta initial condition NaNs the linearising filters; `initial_std` fixes it."""
    field_shape = (1, 4)
    times = jnp.arange(5, dtype=jnp.float64)

    def build(initial_std):
        return spatial_dynamics(
            _decay_stepper(),
            field_shape=field_shape,
            dt_obs=1.0,
            initial_field=jnp.ones(field_shape),
            initial_std=initial_std,
            process_noise_std=0.05,
            observation_noise_std=0.2,
        )

    dynamics = build(0.1)
    sim = dsx.simulate(dynamics, rng_key=jr.PRNGKey(0), predict_times=times)
    assert sim.observations is not None
    obs_values = sim.observations[0]

    def loglik(model):
        with (
            numpyro.handlers.trace() as tr,
            numpyro.handlers.seed(rng_seed=jr.PRNGKey(1)),
        ):
            with dsx.Filter(filter_config=EKFConfig()):
                dsx.sample("f", model, obs_times=times, obs_values=obs_values)
        return tr["f_marginal_loglik"]["value"]

    assert bool(jnp.isfinite(loglik(dynamics)))
    assert bool(jnp.isnan(loglik(build(None))))
