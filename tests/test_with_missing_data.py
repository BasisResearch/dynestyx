import os

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as _dist
import optax
import pytest
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

from dynestyx.discretizers import Discretizer
from dynestyx.models import (
    DiagonalGaussianObservation,
    DiagonalLinearGaussianObservation,
)
from dynestyx.models.core import ObservationModel
from dynestyx.models.observations import DiracIdentityObservation
from dynestyx.models.state_evolution import (
    GaussianStateEvolution,
    LinearGaussianStateEvolution,
)
from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import (
    discrete_time_lti_simplified_model,
    interacting_particles_gaussian_kernel_model,
    particle_sde_gaussian_potential_model,
)
from tests.test_utils import get_output_dir

SAVE_FIG = True
SMOKE = os.environ.get("DYNESTYX_SMOKE_TEST", "0") == "1"


def _sim_site(result: dict, key: str, name: str = "f"):
    return result[key] if key in result else result[f"{name}_{key}"]


def _single_trajectory(result: dict, key: str, name: str = "f"):
    value = _sim_site(result, key, name=name)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def _trajectory_draws(result: dict, key: str, name: str = "f"):
    value = _sim_site(result, key, name=name)
    if value.ndim >= 3 and value.shape[1] == 1:
        return value[:, 0]
    return value


# ---------------------------------------------------------------------------
# Missingness helpers
# ---------------------------------------------------------------------------


def _apply_missingness_pattern(
    obs_values: jnp.ndarray, missingness_pattern: str, missing_key
) -> jnp.ndarray:
    """Apply a row-level or element-level missingness pattern.

    Patterns:
      "none"       – no missingness
      "random"     – ~20% of rows dropped at random
      "sequential" – every 5th row dropped
      "block"      – one contiguous middle block of rows dropped
      "partial"    – ~30% of individual elements NaN (per-dim, not whole-row)
    """
    if missingness_pattern == "none":
        return obs_values

    if missingness_pattern == "partial":
        # ~20% of individual elements missing (independent per-dim, not whole rows).
        mask = jr.bernoulli(missing_key, p=0.8, shape=obs_values.shape)
        return jnp.where(mask, obs_values, jnp.full_like(obs_values, jnp.nan))

    n_obs = obs_values.shape[0]
    keep_mask = jnp.ones((n_obs,), dtype=bool)

    if missingness_pattern == "random":
        keep_mask = jr.bernoulli(missing_key, p=0.8, shape=(n_obs,))
    elif missingness_pattern == "sequential":
        keep_mask = (jnp.arange(n_obs) % 5) != 0
    elif missingness_pattern == "block":
        block_len = max(1, n_obs // 5)
        block_start = (n_obs - block_len) // 2
        block_mask = (jnp.arange(n_obs) >= block_start) & (
            jnp.arange(n_obs) < block_start + block_len
        )
        keep_mask = ~block_mask

    return jnp.where(keep_mask[:, None], obs_values, jnp.full_like(obs_values, jnp.nan))


def _apply_particle_trajectory_missingness(
    obs_values: jnp.ndarray, missingness_type: str, key
) -> jnp.ndarray:
    """Apply per-particle trajectory gaps to (T, N) obs_values.

    Types:
      "none"           – no missingness
      "block"          – each particle loses a contiguous block of length T//20
      "random_segment" – each particle loses a random-length segment
      "partial"        – ~20% of individual elements NaN (per-particle, not whole-row)
    """
    import numpy as np

    obs_np = np.array(obs_values)
    T, N = obs_np.shape

    if missingness_type == "none":
        return jnp.array(obs_np)

    if missingness_type == "partial":
        mask = jr.bernoulli(key, p=0.95, shape=(T, N))
        obs_np = np.where(mask, obs_np, np.nan)
        return jnp.array(obs_np)

    keys = jr.split(key, 2 * N)

    for i in range(N):
        if missingness_type == "block":
            gap_len = max(1, T // 20)
            gap_start = T // 3
        else:  # random_segment
            gap_len = int(
                jr.randint(
                    keys[2 * i],
                    shape=(),
                    minval=max(1, T // 30),
                    maxval=max(2, T // 20),
                )
            )
            gap_start = int(
                jr.randint(keys[2 * i + 1], shape=(), minval=1, maxval=T - gap_len)
            )
        gap_start = max(0, gap_start - 1)
        obs_np[gap_start : gap_start + gap_len, i] = float("nan")

    return jnp.array(obs_np)


# ---------------------------------------------------------------------------
# Test 1: Discrete-time LTI (baseline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_controls", [False, True])
@pytest.mark.parametrize(
    "missingness_pattern",
    ["none", "random", "sequential", "block"],
)
def test_lti_system_missing_data_science(
    use_controls: bool,
    missingness_pattern: str,
):
    """Discrete-time LTI using LTI_discrete factory with missing observations."""
    rng_key = jr.PRNGKey(0)

    data_init_key, mcmc_key, ctrl_key, missing_key = jr.split(rng_key, 4)

    true_alpha = 0.4
    T = 50 if SMOKE else 200
    obs_times = jnp.arange(start=0.0, stop=float(T), step=1.0)

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
            predict_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    obs_values = _single_trajectory(synthetic, "observations")
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

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        plot_times = _single_trajectory(synthetic, "times")
        states = _single_trajectory(synthetic, "states")
        plt.plot(plot_times, states[:, 0], label="x[0]")
        plt.plot(plot_times, states[:, 1], label="x[1]")
        plt.plot(
            plot_times,
            _single_trajectory(synthetic, "observations")[:, 0],
            label="observations",
            linestyle="--",
        )
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    n_mcmc = 50 if SMOKE else 250
    mcmc_key = jr.PRNGKey(0)
    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=n_mcmc,
        num_warmup=n_mcmc,
        progress_bar=False,
    )
    mcmc.run(mcmc_key)

    posterior_alpha = mcmc.get_samples()["alpha"]
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()
    tol = 1.5 if SMOKE else 0.2
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol

    if not SMOKE:
        hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
        hdi_min = hdi_data["x"].sel(hdi="lower").item()
        hdi_max = hdi_data["x"].sel(hdi="higher").item()
        assert hdi_min <= true_alpha <= hdi_max, (
            f"True alpha {true_alpha} not in HDI [{hdi_min}, {hdi_max}]"
        )


# ---------------------------------------------------------------------------
# Test 2: Diagonal observation models with missing data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obs_model_type", ["diagonal_linear_gaussian", "diagonal_gaussian"]
)
@pytest.mark.parametrize(
    "missingness_pattern",
    ["none", "random", "sequential", "block", "partial"],
)
def test_diagonal_obs_missing_data_science(
    obs_model_type: str,
    missingness_pattern: str,
):
    """DiagonalLinearGaussianObservation / DiagonalGaussianObservation with missing data.

    Mirrors test_lti_system_missing_data_science:
    - 'partial' uses masked_log_prob (per-dim NaN, per-step scan path)
    - others use whole-row missingness (row-filter or scan path)
    """
    import dynestyx as dsx

    rng_key = jr.PRNGKey(0)
    data_init_key, mcmc_key, missing_key = jr.split(rng_key, 3)

    true_alpha = 0.4
    T = 50 if SMOKE else 200
    obs_times = jnp.arange(start=0.0, stop=float(T), step=1.0)
    state_dim = 2

    def make_model(obs_times=None, obs_values=None, predict_times=None):
        alpha = numpyro.sample("alpha", _dist.Uniform(-0.7, 0.7))
        if obs_model_type == "diagonal_linear_gaussian":
            state_evo = LinearGaussianStateEvolution(
                A=jnp.array([[alpha, 0.0], [0.0, 0.8]]),
                cov=0.1 * jnp.eye(state_dim),
            )
            obs_model = DiagonalLinearGaussianObservation(
                H=jnp.eye(state_dim), R_diag=jnp.array([0.25, 0.25])
            )
        else:

            def transition(x, u, t_now, t_next):
                return jnp.array([[alpha, 0.0], [0.0, 0.8]]) @ x

            state_evo = GaussianStateEvolution(
                F=transition, cov=0.1 * jnp.eye(state_dim)
            )
            obs_model = DiagonalGaussianObservation(
                h=lambda x, u, t: x,
                R_diag=jnp.array([0.25, 0.25]),
            )
        dynamics = dsx.DynamicalModel(
            initial_condition=_dist.MultivariateNormal(
                jnp.zeros(state_dim), jnp.eye(state_dim)
            ),
            state_evolution=state_evo,
            observation_model=obs_model,
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )

    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        make_model, params=true_params, num_samples=1, exclude_deterministic=False
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(data_init_key, predict_times=obs_times)

    obs_values = _single_trajectory(synthetic, "observations")  # (T, 2)
    obs_values = _apply_missingness_pattern(
        obs_values, missingness_pattern, missing_key
    )

    if missingness_pattern == "partial":
        nan_rows = jnp.isnan(obs_values).any(axis=1)
        full_nan_rows = jnp.isnan(obs_values).all(axis=1)
        assert (nan_rows & ~full_nan_rows).any(), (
            "Expected at least one partial-NaN row"
        )

    OUTPUT_DIR = get_output_dir(
        f"test_diagonal_{obs_model_type}_missing_{missingness_pattern}"
    )

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt
        import numpy as np

        plot_times = np.asarray(_single_trajectory(synthetic, "times"))
        states_np = np.asarray(_single_trajectory(synthetic, "states"))
        obs_np = np.asarray(obs_values)
        fig, axes = plt.subplots(1, state_dim, figsize=(12, 4), sharey=False)
        for dim in range(state_dim):
            axes[dim].plot(plot_times, states_np[:, dim], label=f"x[{dim}]", lw=1.2)
            obs_d = obs_np[:, dim]
            observed = ~np.isnan(obs_d)
            axes[dim].plot(
                plot_times[observed],
                obs_d[observed],
                ".",
                ms=2,
                alpha=0.6,
                color="C1",
                label="obs",
            )
            axes[dim].set_title(f"dim {dim}")
            axes[dim].legend(fontsize=7)
        fig.suptitle(f"{obs_model_type} — {missingness_pattern} missingness")
        fig.tight_layout()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def data_conditioned_model():
        with DiscreteTimeSimulator():
            return make_model(obs_times=obs_times, obs_values=obs_values)

    n_mcmc = 50 if SMOKE else 250
    mcmc = MCMC(
        NUTS(data_conditioned_model),
        num_samples=n_mcmc,
        num_warmup=n_mcmc,
        progress_bar=False,
    )
    mcmc.run(mcmc_key)

    posterior_alpha = mcmc.get_samples()["alpha"]
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()
    tol = 1.5 if SMOKE else 0.25
    assert jnp.abs(posterior_alpha.mean() - true_alpha) < tol, (
        f"alpha not recovered: posterior mean {posterior_alpha.mean():.3f} vs true {true_alpha}"
    )

    if not SMOKE:
        hdi_data = az.hdi(posterior_alpha, hdi_prob=0.95)
        hdi_min = hdi_data["x"].sel(hdi="lower").item()
        hdi_max = hdi_data["x"].sel(hdi="higher").item()
        assert hdi_min <= true_alpha <= hdi_max, (
            f"True alpha {true_alpha} not in HDI [{hdi_min}, {hdi_max}]"
        )


# ---------------------------------------------------------------------------
# Test 3: Particle models (SDE + interacting) with missing data via SVI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unroll_missing", [False, True], ids=["filter", "unroll"])
@pytest.mark.parametrize(
    "model_type, missingness_pattern",
    [
        pytest.param("particle_sde", "none", id="particle_sde-none"),
        pytest.param("particle_sde", "random", id="particle_sde-random"),
        pytest.param("particle_sde", "sequential", id="particle_sde-sequential"),
        pytest.param("particle_sde", "block", id="particle_sde-block"),
        pytest.param("interacting_particles", "none", id="interacting_particles-none"),
        pytest.param(
            "interacting_particles", "block", id="interacting_particles-block"
        ),
        pytest.param(
            "interacting_particles",
            "random_segment",
            id="interacting_particles-random_segment",
        ),
        pytest.param(
            "interacting_particles",
            "partial",
            id="interacting_particles-partial",
        ),
    ],
)
def test_particle_model_missing_data_svi(
    model_type: str,
    missingness_pattern: str,
    unroll_missing: bool,
):
    """Particle models (SDE / interacting) under missing data; parameters recovered via SVI."""
    # particle_sde uses MultivariateNormal transitions (from SDE discretizer).
    # With unroll_missing=True the scan creates latent sample sites at missing
    # rows — too many for SVI to converge. Use MCMC for this combination.
    use_mcmc = (
        model_type == "particle_sde"
        and unroll_missing
        and missingness_pattern != "none"
    ) or missingness_pattern == "partial"

    rng_key = jr.PRNGKey(42)
    data_key, svi_key, missing_key = jr.split(rng_key, 3)

    # ---- Model-specific setup ------------------------------------------------
    num_steps = 500 if SMOKE else 5000

    if model_type == "particle_sde":
        N = 20 if SMOKE else 200
        D, K, sigma = 1, 2, 0.3
        obs_times = jnp.arange(
            start=0.0, stop=5.0 if SMOKE else 10.0, step=0.1 if SMOKE else 0.05
        )
        true_centers = jnp.array([[-2.0], [2.0]])
        true_strengths = jnp.array([1.0, 1.5])
        true_params = {"centers": true_centers, "strengths": true_strengths}
        init_values = {
            "centers": jnp.linspace(-3.0, 3.0, K).reshape(K, D),
            "strengths": 0.5 * jnp.ones(K),
        }

        predictive = Predictive(
            particle_sde_gaussian_potential_model,
            params=true_params,
            num_samples=1,
            exclude_deterministic=False,
        )
        with DiscreteTimeSimulator():
            with Discretizer():
                synthetic = predictive(
                    data_key, N=N, D=D, K=K, sigma=sigma, predict_times=obs_times
                )

        obs_values = _single_trajectory(synthetic, "observations")
        obs_values = _apply_missingness_pattern(
            obs_values, missingness_pattern, missing_key
        )

        def data_conditioned_model():
            with DiscreteTimeSimulator(unroll_missing=unroll_missing):
                with Discretizer():
                    return particle_sde_gaussian_potential_model(
                        N=N,
                        D=D,
                        K=K,
                        sigma=sigma,
                        obs_times=obs_times,
                        obs_values=obs_values,
                    )

    else:  # interacting_particles
        N = 20 if SMOKE else 100
        sigma = 0.2
        obs_times = jnp.arange(start=0.0, stop=3.0 if SMOKE else 5.0, step=0.1)
        bg_centers = jnp.array([[-2.0], [2.0]])
        true_coefficient = -1.0
        true_scale = 1.0
        true_params = {  # type: ignore[assignment]
            "coefficient": jnp.array(true_coefficient),
            "scale": jnp.array(true_scale),
        }
        init_values = {  # type: ignore[assignment]
            "coefficient": jnp.array(true_coefficient),
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
                predict_times=obs_times,
                bg_centers=bg_centers,
            )

        obs_values_clean = _single_trajectory(synthetic, "observations")  # (T, N)
        obs_values = _apply_particle_trajectory_missingness(
            obs_values_clean, missingness_pattern, missing_key
        )

        def data_conditioned_model():  # type: ignore[misc]
            with DiscreteTimeSimulator(unroll_missing=unroll_missing):
                return interacting_particles_gaussian_kernel_model(
                    N=N,
                    sigma=sigma,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    bg_centers=bg_centers,
                )

    # ---- Pre-inference data plot ---------------------------------------------
    OUTPUT_DIR = get_output_dir(f"test_{model_type}_missing_{missingness_pattern}")

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt
        import numpy as np

        plot_times = np.asarray(_single_trajectory(synthetic, "times"))
        states_np = np.asarray(_single_trajectory(synthetic, "states"))
        fig, ax = plt.subplots(figsize=(10, 4))
        n_plot = min(10, states_np.shape[1])
        for i in range(n_plot):
            ax.plot(plot_times, states_np[:, i], alpha=0.5, linewidth=0.5)
        if model_type == "particle_sde":
            for k in range(K):
                ax.axhline(
                    true_centers[k, 0].item(),
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"center {k}={true_centers[k, 0].item():.1f}",
                )
            ax.legend(fontsize=7)
        ax.set_xlabel("time")
        ax.set_ylabel("position")
        ax.set_title(f"{model_type} — first {n_plot} trajectories")
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- SVI inference -------------------------------------------------------
    n_mcmc = 50 if SMOKE else 500
    if use_mcmc:
        mcmc = MCMC(
            NUTS(data_conditioned_model),
            num_warmup=n_mcmc,
            num_samples=n_mcmc,
            progress_bar=False,
        )
        mcmc.run(svi_key)
        posterior_samples = mcmc.get_samples()
    else:
        guide = AutoNormal(
            data_conditioned_model, init_loc_fn=init_to_value(values=init_values)
        )
        optimizer = optax.adam(learning_rate=1e-3)
        svi = SVI(data_conditioned_model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(svi_key, num_steps)

        n_post = 500
        posterior_samples = guide.sample_posterior(
            jr.PRNGKey(1), svi_result.params, sample_shape=(n_post,)
        )

    # ---- Assertions ----------------------------------------------------------
    for name in true_params:
        assert name in posterior_samples
        assert not jnp.isnan(posterior_samples[name]).any(), f"{name} has NaN"
        assert not jnp.isinf(posterior_samples[name]).any(), f"{name} has Inf"

    if not use_mcmc:
        assert svi_result.losses[-1] < svi_result.losses[0], (
            f"SVI loss did not decrease: {svi_result.losses[0]:.1f} → {svi_result.losses[-1]:.1f}"
        )

    if not SMOKE:
        if model_type == "particle_sde":
            tol = 1.0
            # Sort by center[:,0] to handle label switching (MCMC can swap clusters)
            post_centers_mean = posterior_samples["centers"].mean(0)
            sort_idx = jnp.argsort(post_centers_mean[:, 0])
            post_centers_sorted = post_centers_mean[sort_idx]
            true_centers_sorted = true_centers[jnp.argsort(true_centers[:, 0])]
            assert jnp.allclose(post_centers_sorted, true_centers_sorted, atol=tol), (
                f"Centers not recovered: {post_centers_sorted} vs {true_centers_sorted}"
            )
        else:
            tol = 0.3
            post_coeff = posterior_samples["coefficient"]
            post_scale = posterior_samples["scale"]
            assert jnp.abs(post_coeff.mean() - true_coefficient) < tol, (
                f"coefficient not recovered: {post_coeff.mean():.3f} vs {true_coefficient}"
            )
            assert jnp.abs(post_scale.mean() - true_scale) < tol, (
                f"scale not recovered: {post_scale.mean():.3f} vs {true_scale}"
            )

    # ---- Post-inference plots ------------------------------------------------
    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt
        import numpy as np

        if model_type == "particle_sde":
            for k in range(K):
                ax = az.plot_posterior(
                    posterior_samples["centers"][:, k, 0],
                    hdi_prob=0.95,
                    ref_val=true_centers[k, 0].item(),
                )
                ax.set_title(f"center[{k}]")
                plt.savefig(
                    OUTPUT_DIR / f"posterior_center_{k}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()
            for k in range(K):
                az.plot_posterior(
                    posterior_samples["strengths"][:, k],
                    hdi_prob=0.95,
                    ref_val=true_strengths[k].item(),
                )
                plt.savefig(
                    OUTPUT_DIR / f"posterior_strength_{k}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()
        elif not use_mcmc:
            # Trajectory plot with PP imputation + gap shading
            pp_params = guide.sample_posterior(
                jr.PRNGKey(2), svi_result.params, sample_shape=(200,)
            )
            pp_predictive = Predictive(
                interacting_particles_gaussian_kernel_model,
                posterior_samples=pp_params,
            )
            with DiscreteTimeSimulator(unroll_missing=True):
                pp_result = pp_predictive(
                    jr.PRNGKey(3),
                    N=N,
                    sigma=sigma,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    bg_centers=bg_centers,
                )
            pred_states = np.asarray(
                _trajectory_draws(pp_result, "states")
            )  # (200, T, N)
            plot_times_np = np.asarray(obs_times)
            obs_np = np.asarray(obs_values)
            clean_np = np.asarray(obs_values_clean)

            rng_plot = np.random.default_rng(seed=0)
            plot_particles = sorted(rng_plot.choice(N, size=4, replace=False))
            fig, axes_dict = plt.subplot_mosaic(
                [["A", "B"], ["C", "D"], ["E", "E"]],
                figsize=(12, 10),
                height_ratios=[1, 1, 1],
            )
            detail_axes = [axes_dict[k] for k in ["A", "B", "C", "D"]]
            ax_all = axes_dict["E"]
            colors = ["C0", "C1", "C2", "C3"]

            def _shade_missing(ax: plt.Axes, missing: np.ndarray) -> None:
                in_gap = False
                gap_t0 = 0.0
                for t in range(len(plot_times_np)):
                    if missing[t] and not in_gap:
                        gap_t0 = plot_times_np[t]
                        in_gap = True
                    elif not missing[t] and in_gap:
                        ax.axvspan(gap_t0, plot_times_np[t], color="orange", alpha=0.15)
                        in_gap = False
                if in_gap:
                    ax.axvspan(gap_t0, plot_times_np[-1], color="orange", alpha=0.15)

            for (i, ax), col in zip(zip(plot_particles, detail_axes), colors):
                miss = np.isnan(obs_np[:, i])
                _shade_missing(ax, miss)
                ax.plot(
                    plot_times_np,
                    clean_np[:, i],
                    color="black",
                    lw=0.8,
                    ls="--",
                    alpha=0.5,
                    label="True",
                )
                ax.plot(
                    plot_times_np[~miss],
                    obs_np[~miss, i],
                    "o",
                    color=col,
                    ms=2.5,
                    alpha=0.8,
                    label="Obs",
                )
                pp_i = pred_states[:, :, i]
                ax.fill_between(
                    plot_times_np,
                    np.percentile(pp_i, 5, 0),
                    np.percentile(pp_i, 95, 0),
                    color=col,
                    alpha=0.25,
                    label="PP 90% CI",
                )
                ax.plot(plot_times_np, np.median(pp_i, 0), color=col, lw=1.2)
                ax.set_title(f"Particle {i}")
                if i == plot_particles[0]:
                    ax.legend(fontsize=7)

            all_colors = plt.cm.tab20(np.linspace(0, 1, N))  # type: ignore[attr-defined]
            for j in range(N):
                c = all_colors[j]
                pp_med_j = np.median(pred_states[:, :, j], 0)
                miss_j = np.isnan(obs_np[:, j])
                ax_all.plot(plot_times_np, pp_med_j, color=c, lw=0.6, alpha=0.5)
                if miss_j.any():
                    ax_all.plot(
                        plot_times_np[miss_j],
                        pp_med_j[miss_j],
                        "o",
                        color=c,
                        ms=3,
                        mfc="none",
                        markeredgewidth=0.7,
                        alpha=0.8,
                    )
            ax_all.set_title(f"All {N} particles — PP medians; ○ = imputed")
            ax_all.set_xlabel("time")
            fig.suptitle(f"PP imputation — {missingness_pattern}", fontsize=12)
            fig.tight_layout()
            plt.savefig(OUTPUT_DIR / "trajectories.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig2, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(9, 3))
            az.plot_posterior(
                np.asarray(posterior_samples["coefficient"]),
                hdi_prob=0.95,
                ref_val=true_coefficient,
                ax=ax_c,
            )
            ax_c.set_title("coefficient")
            az.plot_posterior(
                np.asarray(posterior_samples["scale"]),
                hdi_prob=0.95,
                ref_val=true_scale,
                ax=ax_s,
            )
            ax_s.set_title("scale")
            fig2.suptitle(f"Posteriors — {missingness_pattern}")
            fig2.tight_layout()
            plt.savefig(OUTPUT_DIR / "posteriors.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)


# ---------------------------------------------------------------------------
# Failure-mode tests
# ---------------------------------------------------------------------------


def test_dirac_masked_log_prob_raises():
    """DiracIdentityObservation.masked_log_prob raises NotImplementedError."""
    obs_model = DiracIdentityObservation()
    y = jnp.ones(2)
    mask = jnp.array([True, False])
    x = jnp.ones(2)
    with pytest.raises(NotImplementedError, match="partial missingness"):
        obs_model.masked_log_prob(y, mask, x)


def test_joint_obs_model_masked_log_prob_raises():
    """ObservationModel.masked_log_prob raises for correlated (non-diagonal) MultivariateNormal."""

    class JointGaussianObs(ObservationModel):
        def __call__(self, x, u, t):
            return _dist.MultivariateNormal(x, jnp.array([[1.0, 0.9], [0.9, 1.0]]))

    obs_model = JointGaussianObs()
    y = jnp.zeros(2)
    mask = jnp.array([True, True])
    x = jnp.zeros(2)
    with pytest.raises(NotImplementedError, match="does not decompose"):
        obs_model.masked_log_prob(y, mask, x)


def test_linear_gaussian_obs_masked_log_prob_raises():
    """LinearGaussianObservation uses full-covariance R and cannot decompose per-dim."""
    from dynestyx.models import LinearGaussianObservation

    obs_model = LinearGaussianObservation(H=jnp.eye(2), R=jnp.eye(2))
    y = jnp.zeros(2)
    mask = jnp.array([True, False])
    x = jnp.zeros(2)
    with pytest.raises(NotImplementedError, match="does not decompose"):
        obs_model.masked_log_prob(y, mask, x)


def test_gaussian_obs_masked_log_prob_raises():
    """GaussianObservation uses full-covariance R and cannot decompose per-dim."""
    from dynestyx.models import GaussianObservation

    obs_model = GaussianObservation(h=lambda x, u, t: x, R=jnp.eye(2))
    y = jnp.zeros(2)
    mask = jnp.array([True, False])
    x = jnp.zeros(2)
    with pytest.raises(NotImplementedError, match="does not decompose"):
        obs_model.masked_log_prob(y, mask, x)


def test_entirely_missing_rows_without_unroll_produces_shorter_output():
    """Without unroll_missing=True, entirely-missing rows are filtered; output is shorter."""
    obs_times = jnp.arange(10.0)
    obs_values_with_gap = jnp.ones((10, 1)).at[4:7, :].set(jnp.nan)

    true_params = {"alpha": jnp.array(0.4)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    with DiscreteTimeSimulator(unroll_missing=False):
        result_filtered = predictive(
            jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values_with_gap
        )
    with DiscreteTimeSimulator():
        result_unrolled = predictive(
            jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values_with_gap
        )

    filtered_states = _single_trajectory(result_filtered, "states")
    unrolled_states = _single_trajectory(result_unrolled, "states")
    unrolled_observations = _single_trajectory(result_unrolled, "observations")
    assert filtered_states.shape[0] == 7
    assert unrolled_states.shape[0] == 10
    assert jnp.isnan(unrolled_observations[4, 0])
