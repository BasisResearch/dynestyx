#!/usr/bin/env python3
"""
LV-UODE debug script — standalone CLI version of lv_uode.ipynb.

Usage:
    python scripts/lv_uode_debug.py [--num-steps 100] [--no-plots] [--no-nuts]
"""
import argparse

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_median, init_to_value

import dynestyx as dsx
from dynestyx import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    Filter,
    LinearGaussianObservation,
    SDESimulator,
)
from dynestyx.inference.filter_configs import ContinuousTimeEKFConfig

# ---------------------------------------------------------------------------
# True system
# ---------------------------------------------------------------------------
_alpha = 1.0    # prey self-growth
_beta  = 0.1    # prey-predator interaction (unknown to model)
_gamma = 0.075  # predator-prey interaction (unknown to model)
_delta = 1.5    # predator self-decay

state_dim = 2   # (x=prey, y=predator)


def lv_true_drift(x):
    prey, pred = x[0], x[1]
    return jnp.array([
        _alpha * prey - _beta * prey * pred,
        _gamma * prey * pred - _delta * pred,
    ])


def lv_known_drift(x):
    prey, pred = x[0], x[1]
    return jnp.array([
        _alpha * prey,
        -_delta * pred,
    ])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sigma_x_true = 0.05
sigma_y_true = 0.5

obs_times = jnp.arange(0.0, 30.0, 0.5)  # 60 observations

H_obs = jnp.array([[1.0, 0.0]])
observation_dim = 1

# Overridden by --observe-both
OBSERVE_BOTH = False

x0_mean = jnp.array([10.0, 5.0])
x0_cov  = 0.5**2 * jnp.eye(state_dim)

initial_condition_kwargs = dict(
    initial_condition=dist.MultivariateNormal(
        loc=x0_mean,
        covariance_matrix=x0_cov,
    )
)

key = jr.PRNGKey(42)
key, k_data, k_svi, k_filter = jr.split(key, 4)


DIFFUSION_FLOOR = 0.0  # overridden by --diffusion-floor
LAPLACE_SCALE = 0.1    # overridden by --laplace-scale
OBS_DT = 0.5           # overridden by --obs-dt

def get_obs_config():
    if OBSERVE_BOTH:
        return jnp.eye(state_dim), state_dim
    else:
        return jnp.array([[1.0, 0.0]]), 1

def make_state_evolution(drift_fn, diffusion_coeff):
    safe_coeff = jnp.maximum(diffusion_coeff, DIFFUSION_FLOOR) if DIFFUSION_FLOOR > 0 else diffusion_coeff
    return ContinuousTimeStateEvolution(
        drift=lambda x, u, t: drift_fn(x),
        diffusion_coefficient=lambda x, u, t: safe_coeff * jnp.eye(state_dim),
    )


# ---------------------------------------------------------------------------
# Polynomial library
# ---------------------------------------------------------------------------
TERM_NAMES = ["1", "x", "y", "x²", "xy", "y²"]
N_TERMS = len(TERM_NAMES)


def interaction_library(x):
    prey, pred = x[0], x[1]
    return jnp.array([1.0, prey, pred, prey**2, prey * pred, pred**2])


true_Theta = jnp.zeros((state_dim, N_TERMS))
true_Theta = true_Theta.at[0, 4].set(-_beta)
true_Theta = true_Theta.at[1, 4].set(_gamma)


# ---------------------------------------------------------------------------
# UODE model
# ---------------------------------------------------------------------------
COV_RESCALING = 1.0  # overridden by --cov-rescaling
FILTER_STATE_ORDER = "first"  # overridden by --filter-state-order

def uode_lv_model(obs_times=None, obs_values=None):
    Theta = numpyro.sample(
        "Theta",
        dist.Laplace(0.0, LAPLACE_SCALE).expand([state_dim, N_TERMS]).to_event(2),
    )
    sigma_x = numpyro.sample("sigma_x", dist.LogNormal(jnp.log(0.05), 1.0))
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.5))

    def drift(x):
        known = lv_known_drift(x)
        phi = interaction_library(x)
        unknown = Theta @ phi
        return known + unknown

    H, obs_dim = get_obs_config()
    return dsx.sample("f", DynamicalModel(
        state_evolution=make_state_evolution(drift, sigma_x),
        observation_model=LinearGaussianObservation(
            H=H,
            R=sigma_y**2 * jnp.eye(obs_dim),
        ),
        **initial_condition_kwargs,
    ), obs_times=obs_times, obs_values=obs_values)


def data_conditioned_model(obs_times=None, obs_values=None):
    with Filter(filter_config=ContinuousTimeEKFConfig(
        record_filtered_states_mean=True,
        record_filtered_states_cov=True,
        cov_rescaling=COV_RESCALING,
        filter_state_order=FILTER_STATE_ORDER,
    )):
        return uode_lv_model(obs_times=obs_times, obs_values=obs_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LV-UODE debug script")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=0.0,
                        help="Gradient clip norm (0=no clip, use Adam)")
    parser.add_argument("--lr-decay", type=float, default=0.0,
                        help="Exponential decay rate (0=no decay)")
    parser.add_argument("--cov-rescaling", type=float, default=1.0)
    parser.add_argument("--filter-state-order", type=str, default="first",
                        choices=["zeroth", "first", "second"])
    parser.add_argument("--diffusion-floor", type=float, default=0.0)
    parser.add_argument("--fwd-mode", action="store_true",
                        help="Use forward_mode_differentiation=True")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-nuts", action="store_true")
    parser.add_argument("--two-phase", action="store_true",
                        help="Phase 1: cov_rescaling=2 fast convergence, "
                             "Phase 2: cov_rescaling=1 to refine sigmas")
    parser.add_argument("--init", type=str, default="prior",
                        choices=["prior", "median", "uninformed", "moderate"],
                        help="Init strategy: prior=prior mode/median (default, principled), "
                             "median=numpyro init_to_median, "
                             "uninformed=sigma_x=0.1, sigma_y=1.0, "
                             "moderate=sigma_x=0.08, sigma_y=0.8")
    parser.add_argument("--laplace-scale", type=float, default=0.1,
                        help="Scale for Laplace prior on Theta (default 0.1)")
    parser.add_argument("--observe-both", action="store_true",
                        help="Observe both prey and predator (H=eye(2))")
    parser.add_argument("--obs-dt", type=float, default=0.5,
                        help="Observation interval (default 0.5)")
    args = parser.parse_args()

    global COV_RESCALING, FILTER_STATE_ORDER, DIFFUSION_FLOOR, OBSERVE_BOTH, LAPLACE_SCALE, OBS_DT
    COV_RESCALING = args.cov_rescaling
    FILTER_STATE_ORDER = args.filter_state_order
    DIFFUSION_FLOOR = args.diffusion_floor
    OBSERVE_BOTH = args.observe_both
    LAPLACE_SCALE = args.laplace_scale
    OBS_DT = args.obs_dt
    obs_times = jnp.arange(0.0, 30.0, OBS_DT)

    # --- Data generation ---
    print("=" * 60)
    print("Generating synthetic data …")
    print("=" * 60)

    def model_with_true_drift(obs_times=None, obs_values=None):
        H, obs_dim = get_obs_config()
        return dsx.sample("f", DynamicalModel(
            state_evolution=make_state_evolution(lv_true_drift, sigma_x_true),
            observation_model=LinearGaussianObservation(
                H=H,
                R=sigma_y_true**2 * jnp.eye(obs_dim),
            ),
            **initial_condition_kwargs,
        ), obs_times=obs_times, obs_values=obs_values)

    predictive = Predictive(model_with_true_drift, num_samples=1, exclude_deterministic=False)
    with SDESimulator(dt0=1e-3):
        synthetic = predictive(k_data, obs_times=obs_times)

    obs_values = synthetic["observations"][0]
    states = synthetic["states"][0]

    print(f"  obs_values shape: {obs_values.shape}")
    print(f"  states shape:     {states.shape}")

    print("\nTrue Theta:")
    print(f"  Terms: {TERM_NAMES}")
    print(f"  {true_Theta}")

    # --- SVI ---
    print("\n" + "=" * 60)
    print(f"Running SVI ({args.num_steps} steps) …")
    print("=" * 60)

    print(f"  cov_rescaling: {args.cov_rescaling}")
    print(f"  filter_state_order: {args.filter_state_order}")
    print(f"  forward_mode_diff: {args.fwd_mode}")
    print(f"  init: {args.init}")
    print(f"  two_phase: {args.two_phase}")
    if args.init == "prior":
        # Principled: prior median for sigma_x (LogNormal median = exp(loc)),
        # prior scale for sigma_y (HalfNormal scale), prior mode for Theta (Laplace mode = 0)
        init_fn = init_to_value(values={
            "sigma_x": jnp.array(0.05),   # = exp(log(0.05)) = prior median
            "sigma_y": jnp.array(0.5),    # = HalfNormal scale
            "Theta": jnp.zeros((state_dim, N_TERMS)),  # = Laplace mode
        })
    elif args.init == "median":
        init_fn = init_to_median()
    elif args.init == "uninformed":
        init_fn = init_to_value(values={
            "sigma_x": jnp.array(0.1),
            "sigma_y": jnp.array(1.0),
            "Theta": jnp.zeros((state_dim, N_TERMS)),
        })
    elif args.init == "moderate":
        init_fn = init_to_value(values={
            "sigma_x": jnp.array(0.08),
            "sigma_y": jnp.array(0.8),
            "Theta": jnp.zeros((state_dim, N_TERMS)),
        })
    guide = AutoDelta(
        data_conditioned_model,
        init_loc_fn=init_fn,
    )

    def run_svi_phase(guide, num_steps, lr, lr_decay, clip_norm, cov_rescaling, label="SVI"):
        """Run one phase of SVI, returning (svi_result, losses_array)."""
        global COV_RESCALING
        COV_RESCALING = cov_rescaling

        if clip_norm > 0:
            optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=clip_norm)
            opt_str = f"ClippedAdam(lr={lr}, clip_norm={clip_norm})"
        elif lr_decay > 0:
            import optax
            schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=max(1, num_steps // 5),
                decay_rate=lr_decay,
            )
            optimizer = numpyro.optim.optax_to_numpyro(optax.adam(schedule))
            opt_str = f"Adam(lr={lr}, exp_decay={lr_decay}, transition={num_steps // 5})"
        else:
            optimizer = numpyro.optim.Adam(step_size=lr)
            opt_str = f"Adam(lr={lr})"

        print(f"\n  [{label}] {num_steps} steps | {opt_str} | cov_rescaling={cov_rescaling}")

        svi = SVI(data_conditioned_model, guide, optimizer, loss=Trace_ELBO())
        svi_kwargs = dict(
            obs_times=obs_times, obs_values=obs_values,
            progress_bar=True, stable_update=True,
        )
        if args.fwd_mode:
            svi_kwargs["forward_mode_differentiation"] = True
        result = svi.run(k_svi, num_steps, **svi_kwargs)
        return result

    if args.two_phase:
        # Phase 1: cov_rescaling=2.0 for fast, stable Theta convergence
        phase1_steps = args.num_steps // 2
        svi_result = run_svi_phase(
            guide, phase1_steps, lr=args.lr, lr_decay=args.lr_decay,
            clip_norm=args.clip_norm, cov_rescaling=2.0, label="Phase 1 (rescaled)",
        )
        # Warm-start phase 2 from phase 1 MAP
        median_p1 = guide.median(svi_result.params)
        guide_p2 = AutoDelta(
            data_conditioned_model,
            init_loc_fn=init_to_value(values={
                "sigma_x": median_p1["sigma_x"],
                "sigma_y": median_p1["sigma_y"],
                "Theta": median_p1["Theta"],
            }),
        )
        # Phase 2: cov_rescaling=1.0 for unbiased sigma refinement
        phase2_steps = args.num_steps - phase1_steps
        svi_result = run_svi_phase(
            guide_p2, phase2_steps, lr=args.lr * 0.3, lr_decay=max(args.lr_decay, 0.5),
            clip_norm=args.clip_norm, cov_rescaling=1.0, label="Phase 2 (refine)",
        )
        guide = guide_p2
    else:
        svi_result = run_svi_phase(
            guide, args.num_steps, lr=args.lr, lr_decay=args.lr_decay,
            clip_norm=args.clip_norm, cov_rescaling=args.cov_rescaling, label="SVI",
        )

    median = guide.median(svi_result.params)
    Theta_inferred = median["Theta"]
    sigma_x_inferred = float(median["sigma_x"])
    sigma_y_inferred = float(median["sigma_y"])

    print(f"\nInferred sigma_x = {sigma_x_inferred:.4f}  (true = {sigma_x_true:.4f})")
    print(f"Inferred sigma_y = {sigma_y_inferred:.4f}  (true = {sigma_y_true:.4f})")

    print("\nTheta recovery:")
    print(f"{'':>12s}  {'x-row (prey drift)':>22s}  {'y-row (predator drift)':>24s}")
    print(f"{'':>12s}  {'true':>8s} {'inferred':>10s}  {'true':>8s} {'inferred':>10s}")
    print("-" * 68)
    for j, name in enumerate(TERM_NAMES):
        t0, t1 = float(true_Theta[0, j]), float(true_Theta[1, j])
        i0, i1 = float(Theta_inferred[0, j]), float(Theta_inferred[1, j])
        print(f"{name:>12s}  {t0:>8.4f} {i0:>10.4f}  {t1:>8.4f} {i1:>10.4f}")

    # Theta error summary
    import numpy as np
    err = np.asarray(Theta_inferred - true_Theta)
    print(f"\nTheta error: Frobenius={np.linalg.norm(err):.4f}  "
          f"MAE={np.abs(err).mean():.4f}  max|err|={np.abs(err).max():.4f}")
    # Sparsity: fraction of entries < 0.01 (should be 10/12)
    n_sparse = (np.abs(np.asarray(Theta_inferred)) < 0.01).sum()
    print(f"Sparsity: {n_sparse}/{Theta_inferred.size} entries <0.01  "
          f"(true: 10/{Theta_inferred.size} are zero)")
    # xy term accuracy
    xy_prey_err = float(Theta_inferred[0, 4] - true_Theta[0, 4])
    xy_pred_err = float(Theta_inferred[1, 4] - true_Theta[1, 4])
    print(f"xy term: prey err={xy_prey_err:+.4f}  pred err={xy_pred_err:+.4f}")

    # Check for NaN in losses
    import numpy as np
    losses = np.asarray(svi_result.losses)
    n_nan = np.isnan(losses).sum()
    first_nan = int(np.argmax(np.isnan(losses))) if n_nan > 0 else None
    print(f"\nLoss: init={losses[0]:.2f}  final={losses[-1]:.2f}  "
          f"min={np.nanmin(losses):.2f}  NaN count={n_nan}"
          + (f"  first NaN at step {first_nan}" if first_nan is not None else ""))

    # Save losses for external analysis
    np.save("lv_uode_losses.npy", losses)
    print("Saved lv_uode_losses.npy")

    # --- Tail ELBO diagnostics ---
    n_tail = min(500, len(losses) // 4)
    if n_tail >= 50:
        tail = losses[-n_tail:]
        tail_clean = tail[~np.isnan(tail)]
        if len(tail_clean) > 20:
            print(f"\n--- Tail ELBO analysis (last {n_tail} steps) ---")
            print(f"  mean={tail_clean.mean():.2f}  std={tail_clean.std():.2f}  "
                  f"min={tail_clean.min():.2f}  max={tail_clean.max():.2f}")
            print(f"  range={tail_clean.max()-tail_clean.min():.2f}  "
                  f"CoV={tail_clean.std()/abs(tail_clean.mean())*100:.2f}%")
            # Autocorrelation for periodicity detection
            tc = tail_clean - tail_clean.mean()
            acf = np.correlate(tc, tc, mode="full")
            acf = acf[len(acf)//2:]
            acf /= acf[0] + 1e-30
            from scipy.signal import find_peaks
            peaks, props = find_peaks(acf[1:], height=0.05)
            if len(peaks) > 0:
                top = sorted(zip(props["peak_heights"], peaks+1), reverse=True)[:5]
                acf_desc = ", ".join(f"lag={l} r={h:.3f}" for h, l in top)
                print(f"  ACF peaks: [{acf_desc}]")
                print(f"  Dominant period ~ {top[0][1]} steps")
            else:
                print("  No clear periodicity in autocorrelation")
            # Print every 50th in last 200
            print(f"  Sample losses (every 50th, last 200):")
            start = max(0, len(losses) - 200)
            for i in range(start, len(losses), 50):
                print(f"    step {i}: {losses[i]:.2f}")
            print(f"    step {len(losses)-1}: {losses[-1]:.2f}")

    # --- NUTS ---
    if not args.no_nuts:
        print("\n" + "=" * 60)
        print("Running NUTS …")
        print("=" * 60)

        from numpyro.infer import MCMC, NUTS

        k_mcmc = jr.PRNGKey(99)
        nuts_kernel = NUTS(
            data_conditioned_model,
            init_strategy=init_to_value(values=median),
        )
        mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=200, num_chains=1,
                     progress_bar=True)
        mcmc.run(k_mcmc, obs_times=obs_times, obs_values=obs_values)
        mcmc.print_summary()
        posterior = mcmc.get_samples()

        Theta_posterior = posterior["Theta"]
        Theta_nuts_mean = Theta_posterior.mean(axis=0)
        Theta_nuts_std = Theta_posterior.std(axis=0)

        print(f"\n{'':>8s}  {'── prey (x) drift ──':>28s}    {'── predator (y) drift ──':>30s}")
        print(f"{'term':>8s}  {'true':>6s} {'MAP':>8s} {'NUTS':>12s}    {'true':>6s} {'MAP':>8s} {'NUTS':>12s}")
        print("-" * 88)
        for j, name in enumerate(TERM_NAMES):
            t0 = float(true_Theta[0, j]); t1 = float(true_Theta[1, j])
            m0 = float(Theta_inferred[0, j]); m1 = float(Theta_inferred[1, j])
            n0 = float(Theta_nuts_mean[0, j]); n1 = float(Theta_nuts_mean[1, j])
            s0 = float(Theta_nuts_std[0, j]); s1 = float(Theta_nuts_std[1, j])
            print(f"{name:>8s}  {t0:>6.3f} {m0:>8.4f} {n0:>7.4f}±{s0:.4f}    {t1:>6.3f} {m1:>8.4f} {n1:>7.4f}±{s1:.4f}")

    # --- Plots ---
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(losses, lw=0.8, color="steelblue")
        ax.set_xlabel("SVI step")
        ax.set_ylabel("ELBO loss")
        ax.set_title("SVI convergence")
        ax.set_yscale("log")
        plt.tight_layout()
        fig.savefig("lv_uode_loss.png", dpi=150)
        print("\nSaved lv_uode_loss.png")

        # Theta heatmap
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for ax, mat, title in zip(axes,
                                   [true_Theta, Theta_inferred],
                                   ["True Theta", "Inferred Theta (MAP)"]):
            vmax = float(jnp.abs(mat).max()) + 1e-6
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(TERM_NAMES)))
            ax.set_xticklabels(TERM_NAMES, rotation=35, ha="right", fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["prey (x)", "predator (y)"])
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        fig.savefig("lv_uode_theta.png", dpi=150)
        print("Saved lv_uode_theta.png")

        # Filtered trajectories
        predictive_filter = Predictive(data_conditioned_model, params=svi_result.params, num_samples=1)
        pred_samples = predictive_filter(k_filter, obs_times=obs_times, obs_values=obs_values)
        filtered_mean = pred_samples["f_filtered_states_mean"][0]
        filtered_cov = pred_samples["f_filtered_states_cov"][0]
        filtered_std = jnp.sqrt(jnp.diagonal(filtered_cov, axis1=-2, axis2=-1))

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for i, (ax, label, col_t, col_f) in enumerate(zip(
            axes, ["Prey (x)", "Predator (y)"],
            ["tab:blue", "tab:orange"], ["steelblue", "darkorange"],
        )):
            ax.plot(obs_times, states[:, i], lw=1.5, color=col_t, label="True state", zorder=3)
            mu = filtered_mean[:, i]
            std = filtered_std[:, i]
            ax.plot(obs_times, mu, "--", lw=1.5, color=col_f, label="Filter mean", zorder=4)
            ax.fill_between(obs_times, mu - std, mu + std, color=col_f, alpha=0.25, label="±1σ")
            if i == 0:
                ax.scatter(obs_times, obs_values[:, 0], s=12, color="black", zorder=5, label="Observations")
            ax.set_ylabel(label)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time")
        axes[0].set_title("EKF-filtered trajectories (MAP parameters)")
        plt.tight_layout()
        fig.savefig("lv_uode_filtered.png", dpi=150)
        print("Saved lv_uode_filtered.png")

        plt.close("all")

    print("\nDone.")


if __name__ == "__main__":
    main()
