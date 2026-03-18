import arviz as az
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

from dynestyx.discretizers import Discretizer
from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import (
    discrete_time_lti_simplified_model,
    interacting_particles_gaussian_kernel_model,
    particle_sde_gaussian_potential_model,
)
from tests.test_utils import get_output_dir

SAVE_FIG = True


def _apply_missingness_pattern(
    obs_values: jnp.ndarray, missingness_pattern: str, missing_key
) -> jnp.ndarray:
    if missingness_pattern == "none":
        return obs_values

    n_obs = obs_values.shape[0]
    missing_values = jnp.full_like(obs_values, jnp.nan)
    keep_mask = jnp.ones((n_obs,), dtype=bool)

    if missingness_pattern == "random":
        # Randomly drop roughly 20% of observations.
        keep_mask = jr.bernoulli(missing_key, p=0.8, shape=(n_obs,))
    elif missingness_pattern == "sequential":
        # Regularly drop every 5th observation.
        keep_mask = (jnp.arange(n_obs) % 5) != 0
    elif missingness_pattern == "block":
        # Drop one contiguous middle block.
        block_len = max(1, n_obs // 5)
        block_start = (n_obs - block_len) // 2
        block_mask = (jnp.arange(n_obs) >= block_start) & (
            jnp.arange(n_obs) < block_start + block_len
        )
        keep_mask = ~block_mask

    return jnp.where(keep_mask[:, None], obs_values, missing_values)


@pytest.mark.parametrize("use_controls", [False, True])
@pytest.mark.parametrize(
    "missingness_pattern",
    ["none", "random", "sequential", "block"],
)
@pytest.mark.parametrize("num_samples", [250])
def test_lti_system_missing_data_science(
    use_controls: bool,
    missingness_pattern: str,
    num_samples: int,
):
    """Discrete-time LTI using LTI_discrete factory with missing observations."""
    rng_key = jr.PRNGKey(0)

    data_init_key, mcmc_key, ctrl_key, missing_key = jr.split(rng_key, 4)

    true_alpha = 0.4
    # Longer timeseries (~200 obs) so data inform alpha more, like continuous LTI
    obs_times = jnp.arange(start=0.0, stop=200.0, step=1.0)

    ctrl_times = None
    ctrl_values = None
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        ctrl_times = obs_times

    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(
            data_init_key,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    obs_values = synthetic["observations"].squeeze(0)
    obs_values = _apply_missingness_pattern(
        obs_values, missingness_pattern, missing_key
    )

    def data_conditioned_model():
        with DiscreteTimeSimulator():
            return discrete_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

    output_dir_name = (
        "test_lti_discrete_simplified"
        + ("_controlled" if use_controls else "")
        + f"_missing_{missingness_pattern}"
    )
    OUTPUT_DIR = get_output_dir(output_dir_name)

    plot_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        plt.plot(
            plot_times.squeeze(0),
            synthetic["states"].squeeze(0)[:, 0],
            label="x[0]",
        )
        plt.plot(
            plot_times.squeeze(0),
            synthetic["states"].squeeze(0)[:, 1],
            label="x[1]",
        )
        plt.plot(
            plot_times.squeeze(0),
            synthetic["observations"].squeeze(0)[:, 0],
            label="observations",
            linestyle="--",
        )
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    mcmc_key = jr.PRNGKey(0)
    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )

    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    assert "alpha" in posterior_samples
    posterior_alpha = posterior_samples["alpha"]
    assert len(posterior_alpha) == num_samples
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        az.plot_posterior(
            posterior_alpha, hdi_prob=0.95, ref_val=true_params["alpha"].item()
        )
        plt.savefig(OUTPUT_DIR / "posterior_alpha.png", dpi=150, bbox_inches="tight")
        plt.close()

    true_alpha = true_params["alpha"].item()
    tol = 0.2
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol

    hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_alpha <= hdi_max, (
        f"True alpha {true_alpha} not in HDI {hdi_min}, {hdi_max}"
    )


@pytest.mark.parametrize(
    "missingness_pattern",
    ["none", "random", "sequential", "block"],
)
@pytest.mark.parametrize("num_steps", [5000])
def test_particle_sde_missing_data_svi(
    missingness_pattern: str,
    num_steps: int,
):
    """Particle SDE with gradient-of-Gaussian potential: infer centers and strengths."""
    rng_key = jr.PRNGKey(42)
    data_init_key, svi_key, missing_key = jr.split(rng_key, 3)

    N, D, K, sigma = 200, 1, 2, 0.3
    obs_times = jnp.arange(start=0.0, stop=10.0, step=0.05)

    true_centers = jnp.array([[-2.0], [2.0]])  # sorted for label-switching symmetry
    true_strengths = jnp.array([1.0, 1.5])
    true_params = {"centers": true_centers, "strengths": true_strengths}

    # --- generate synthetic data ---
    predictive = Predictive(
        particle_sde_gaussian_potential_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        with Discretizer():
            synthetic = predictive(
                data_init_key,
                N=N,
                D=D,
                K=K,
                sigma=sigma,
                obs_times=obs_times,
            )

    obs_values = synthetic["observations"].squeeze(0)
    obs_values = _apply_missingness_pattern(
        obs_values, missingness_pattern, missing_key
    )

    # --- conditioned model ---
    def data_conditioned_model():
        with DiscreteTimeSimulator():
            with Discretizer():
                return particle_sde_gaussian_potential_model(
                    N=N,
                    D=D,
                    K=K,
                    sigma=sigma,
                    obs_times=obs_times,
                    obs_values=obs_values,
                )

    output_dir_name = f"test_particle_sde_missing_{missingness_pattern}"
    OUTPUT_DIR = get_output_dir(output_dir_name)

    plot_times = synthetic["times"].squeeze(0)

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        states = synthetic["states"].squeeze(0)  # (T, N*D)
        for i in range(min(N, 10)):
            plt.plot(plot_times, states[:, i * D], alpha=0.5, linewidth=0.5)
        for k in range(K):
            plt.axhline(
                true_centers[k, 0].item(),
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"center {k} = {true_centers[k, 0].item():.1f}",
            )
        plt.xlabel("time")
        plt.ylabel("position")
        plt.title("Particle trajectories (first 10)")
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- SVI inference with AutoNormal guide ---
    # Initialize centers with spread to break label-switching symmetry
    init_values = {
        "centers": jnp.linspace(-3.0, 3.0, K).reshape(K, D),
        "strengths": 0.5 * jnp.ones(K),
    }
    guide = AutoNormal(
        data_conditioned_model, init_loc_fn=init_to_value(values=init_values)
    )
    optimizer = optax.adam(learning_rate=1e-3)
    svi = SVI(data_conditioned_model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(svi_key, num_steps)

    num_samples = 500
    posterior_samples = guide.sample_posterior(
        jr.PRNGKey(1), svi_result.params, sample_shape=(num_samples,)
    )

    # --- assertions ---
    assert "centers" in posterior_samples
    assert "strengths" in posterior_samples

    posterior_centers = posterior_samples["centers"]
    posterior_strengths = posterior_samples["strengths"]

    assert not jnp.isnan(posterior_centers).any()
    assert not jnp.isinf(posterior_centers).any()
    assert not jnp.isnan(posterior_strengths).any()
    assert not jnp.isinf(posterior_strengths).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        init_centers = init_values["centers"]

        for k in range(K):
            ax = az.plot_posterior(
                posterior_centers[:, k, 0],
                hdi_prob=0.95,
                ref_val=true_centers[k, 0].item(),
            )
            ax.axvline(
                init_centers[k, 0].item(),
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"init = {init_centers[k, 0].item():.2f}",
            )
            ax.legend()
            plt.savefig(
                OUTPUT_DIR / f"posterior_center_{k}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        init_strengths = init_values["strengths"]

        for k in range(K):
            ax = az.plot_posterior(
                posterior_strengths[:, k],
                hdi_prob=0.95,
                ref_val=true_strengths[k].item(),
            )
            ax.axvline(
                init_strengths[k].item(),
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"init = {init_strengths[k].item():.2f}",
            )
            ax.legend()
            plt.savefig(
                OUTPUT_DIR / f"posterior_strength_{k}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

    # Compare posterior means to true values
    mean_centers = posterior_centers.mean(axis=0)
    mean_strengths = posterior_strengths.mean(axis=0)

    centers_tol = 1.0
    strengths_tol = 1.0

    assert jnp.allclose(mean_centers, true_centers, atol=centers_tol), (
        f"Centers not recovered: {mean_centers} vs {true_centers}"
    )

    assert jnp.allclose(mean_strengths, true_strengths, atol=strengths_tol), (
        f"Strengths not recovered: {mean_strengths} vs {true_strengths}"
    )

    # Check true values fall within 99% HDI
    for k in range(K):
        for d in range(D):
            hdi_data = az.hdi(posterior_centers[:, k, d], hdi_prob=0.99)
            hdi_min = hdi_data["x"].sel(hdi="lower").item()
            hdi_max = hdi_data["x"].sel(hdi="higher").item()
            assert hdi_min <= true_centers[k, d] <= hdi_max, (
                f"True center[{k},{d}]={true_centers[k, d]} not in HDI [{hdi_min}, {hdi_max}]"
            )

    for k in range(K):
        hdi_data = az.hdi(posterior_strengths[:, k], hdi_prob=0.99)
        hdi_min = hdi_data["x"].sel(hdi="lower").item()
        hdi_max = hdi_data["x"].sel(hdi="higher").item()
        assert hdi_min <= true_strengths[k] <= hdi_max, (
            f"True strength[{k}]={true_strengths[k]} not in HDI [{hdi_min}, {hdi_max}]"
        )


# ---------------------------------------------------------------------------
# Helpers for partial / per-particle trajectory missingness
# ---------------------------------------------------------------------------


def _apply_partial_missingness_pattern(
    obs_values: jnp.ndarray, frac_missing: float, key
) -> jnp.ndarray:
    """Randomly NaN individual obs dimensions (not whole rows)."""
    mask = jr.bernoulli(key, p=1.0 - frac_missing, shape=obs_values.shape)
    return jnp.where(mask, obs_values, jnp.nan)


def _apply_particle_trajectory_missingness(
    obs_values: jnp.ndarray, missingness_type: str, key
) -> jnp.ndarray:
    """Apply per-particle trajectory gaps (and optional ID switch) to (T, N) obs_values.

    Types:
      "none"              – no missingness
      "block"             – each particle loses a contiguous block of length T//4
      "random_segment"    – each particle loses a random-length segment
      "block_id_switch"   – block gap + swap columns 0,1 after the gap
      "random_id_switch"  – random_segment + swap columns 0,1 after the gap
    """
    import numpy as np

    obs_np = np.array(obs_values)
    T, N = obs_np.shape

    if missingness_type == "none":
        return jnp.array(obs_np)

    keys = jr.split(key, 2 * N)
    gap_ends = []

    for i in range(N):
        if "block" in missingness_type:
            gap_len = max(1, T // 20)
            gap_start = T // 3
        else:  # random_segment
            gap_len = int(
                jr.randint(
                    keys[2 * i],
                    shape=(),
                    minval=max(1, T // 30),
                    maxval=max(2, T // 10),
                )
            )
            gap_start = int(
                jr.randint(keys[2 * i + 1], shape=(), minval=0, maxval=T - gap_len)
            )
        gap_end = gap_start + gap_len
        obs_np[gap_start:gap_end, i] = float("nan")
        gap_ends.append(gap_end)

    if "id_switch" in missingness_type:
        # After both particles 0 and 1 have resumed, swap their columns
        switch_t = max(gap_ends[0], gap_ends[1])
        col0 = obs_np[switch_t:, 0].copy()
        obs_np[switch_t:, 0] = obs_np[switch_t:, 1]
        obs_np[switch_t:, 1] = col0

    return jnp.array(obs_np)


# ---------------------------------------------------------------------------
# Test A: LTI with per-dim partial missingness (not whole rows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_samples", [250])
def test_lti_partial_missingness(num_samples: int):
    """LTI system with partial (per-dim) NaN obs → two-mask scan → posterior recovery."""
    rng_key = jr.PRNGKey(7)
    data_key, mcmc_key, missing_key = jr.split(rng_key, 3)

    true_alpha = 0.4
    obs_times = jnp.arange(start=0.0, stop=100.0, step=1.0)

    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(data_key, obs_times=obs_times)

    obs_values = synthetic["observations"].squeeze(0)
    obs_values = _apply_partial_missingness_pattern(
        obs_values, frac_missing=0.3, key=missing_key
    )

    def data_conditioned_model():
        with DiscreteTimeSimulator():
            return discrete_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()

    assert "alpha" in posterior_samples
    posterior_alpha = posterior_samples["alpha"]
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()

    tol = 0.2
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol

    hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    assert hdi_min <= true_alpha <= hdi_max, (
        f"True alpha {true_alpha} not in HDI [{hdi_min}, {hdi_max}]"
    )


# ---------------------------------------------------------------------------
# Test B: LTI with full-row missing + unroll_missing=True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_samples", [250])
def test_lti_unroll_missing_rows(num_samples: int):
    """LTI with block-missing rows + unroll_missing=True → full-length state output + posterior."""
    rng_key = jr.PRNGKey(11)
    data_key, mcmc_key, missing_key = jr.split(rng_key, 3)

    true_alpha = 0.4
    obs_times = jnp.arange(start=0.0, stop=100.0, step=1.0)
    T = len(obs_times)

    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(data_key, obs_times=obs_times)

    obs_values = synthetic["observations"].squeeze(0)
    # Apply block full-row missingness
    obs_values = _apply_missingness_pattern(obs_values, "block", missing_key)

    # With unroll_missing=True, the output should preserve all T time steps
    with DiscreteTimeSimulator(unroll_missing=True):
        predictive_unroll = Predictive(
            discrete_time_lti_simplified_model,
            params=true_params,
            num_samples=1,
            exclude_deterministic=False,
        )
        result = predictive_unroll(data_key, obs_times=obs_times, obs_values=obs_values)

    assert result["states"].shape[1] == T, (
        f"Expected T={T} time steps in states, got {result['states'].shape[1]}"
    )
    # Missing rows should be NaN in observations output
    block_len = max(1, T // 5)
    block_start = (T - block_len) // 2
    assert jnp.isnan(result["observations"].squeeze(0)[block_start, 0]), (
        "Expected NaN in observations at missing rows"
    )

    def data_conditioned_model():
        with DiscreteTimeSimulator(unroll_missing=True):
            return discrete_time_lti_simplified_model(
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=num_samples,
        num_warmup=num_samples,
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()

    assert "alpha" in posterior_samples
    posterior_alpha = posterior_samples["alpha"]
    assert not jnp.isnan(posterior_alpha).any()

    tol = 0.25
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol


# ---------------------------------------------------------------------------
# Test C: Interacting particles with Gaussian kernel + per-particle trajectory gaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missingness_type",
    ["none", "block", "random_segment", "block_id_switch", "random_id_switch"],
)
@pytest.mark.parametrize("num_steps", [3000])
def test_interacting_particles_partial_missingness_svi(
    missingness_type: str,
    num_steps: int,
):
    """Interacting particles with Gaussian kernel: infer loc and scale under trajectory gaps."""
    rng_key = jr.PRNGKey(42)
    data_key, svi_key, missing_key = jr.split(rng_key, 3)

    N = 4
    sigma = 0.2
    obs_times = jnp.arange(start=0.0, stop=5.0, step=0.1)

    true_loc = 0.5
    true_scale = 0.5
    true_params = {
        "loc": jnp.array(true_loc),
        "scale": jnp.array(true_scale),
    }

    predictive = Predictive(
        interacting_particles_gaussian_kernel_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator():
        synthetic = predictive(
            data_key,
            N=N,
            sigma=sigma,
            obs_times=obs_times,
        )

    obs_values = synthetic["observations"].squeeze(0)  # (T, N)
    obs_values = _apply_particle_trajectory_missingness(
        obs_values, missingness_type, missing_key
    )

    def data_conditioned_model():
        with DiscreteTimeSimulator(unroll_missing=True):
            return interacting_particles_gaussian_kernel_model(
                N=N,
                sigma=sigma,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    init_values = {
        "loc": jnp.array(0.0),
        "scale": jnp.array(1.0),
    }
    guide = AutoNormal(
        data_conditioned_model, init_loc_fn=init_to_value(values=init_values)
    )
    optimizer = optax.adam(learning_rate=1e-3)
    svi = SVI(data_conditioned_model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(svi_key, num_steps)

    num_posterior_samples = 500
    posterior_samples = guide.sample_posterior(
        jr.PRNGKey(1), svi_result.params, sample_shape=(num_posterior_samples,)
    )

    assert "loc" in posterior_samples
    assert "scale" in posterior_samples

    posterior_loc = posterior_samples["loc"]
    posterior_scale = posterior_samples["scale"]

    assert not jnp.isnan(posterior_loc).any()
    assert not jnp.isinf(posterior_loc).any()
    assert not jnp.isnan(posterior_scale).any()
    assert not jnp.isinf(posterior_scale).any()

    tol = 0.3
    assert jnp.abs(posterior_loc.mean() - true_loc) < tol, (
        f"loc not recovered: posterior mean {posterior_loc.mean():.3f} vs true {true_loc}"
    )
    assert jnp.abs(posterior_scale.mean() - true_scale) < tol, (
        f"scale not recovered: posterior mean {posterior_scale.mean():.3f} vs true {true_scale}"
    )
