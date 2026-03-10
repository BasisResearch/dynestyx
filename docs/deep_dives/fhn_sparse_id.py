"""Sparse identification of FitzHugh-Nagumo dynamics with multiple inference modes.

This script mirrors docs/deep_dives/fhn_sparse_id.ipynb and extends it by:
1) fitting Laplace-prior and Horseshoe-prior sparse models,
2) running inference with MAP (AutoDelta), SVI+AutoNormal, and/or MCMC (NUTS), and
3) reporting formal quantitative evaluation metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

import dynestyx as dsx
from dynestyx import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    Filter,
    LinearGaussianObservation,
    SDESimulator,
)


STATE_DIM = 2
OBS_DIM = 2
TERM_NAMES = ["1", "v", "w", "v^2", "vw", "w^2", "v^3", "v^2w", "vw^2", "w^3"]
N_TERMS = len(TERM_NAMES)
EPS = 1e-8

# FitzHugh-Nagumo parameters
_A, _B, _C, _I = 0.08, 0.7, 0.8, 0.5

# Data-generation truth
SIGMA_X_TRUE = 0.01
SIGMA_Y_TRUE = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tmax", type=float, default=50.0)

    # MAP / AutoNormal SVI controls
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--step-size", type=float, default=1e-2)

    # MCMC controls
    parser.add_argument("--mcmc-warmup", type=int, default=400)
    parser.add_argument("--mcmc-samples", type=int, default=600)
    parser.add_argument("--mcmc-chains", type=int, default=1)

    # Posterior controls
    parser.add_argument("--posterior-samples", type=int, default=200)
    parser.add_argument("--filter-eval-samples", type=int, default=100)

    parser.add_argument("--support-threshold", type=float, default=0.05)
    parser.add_argument("--inclusion-prob-threshold", type=float, default=0.5)

    parser.add_argument(
        "--methods",
        type=str,
        default="map,autonormal,mcmc",
        help="comma-separated subset of: map,autonormal,mcmc",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default="laplace,horseshoe",
        help="comma-separated subset of: laplace,horseshoe",
    )
    return parser.parse_args()


def fitzhugh_nagumo_drift(x: jnp.ndarray) -> jnp.ndarray:
    v, w = x[0], x[1]
    dv = v - (1.0 / 3.0) * v**3 - w + _I
    dw = _A * (v + _B - _C * w)
    return jnp.array([dv, dw])


def monomials(x: jnp.ndarray) -> jnp.ndarray:
    v, w = x[0], x[1]
    return jnp.array([1.0, v, w, v**2, v * w, w**2, v**3, v**2 * w, v * w**2, w**3])


def true_theta() -> jnp.ndarray:
    theta = jnp.zeros((STATE_DIM, N_TERMS))
    theta = theta.at[0, 0].set(_I)
    theta = theta.at[0, 1].set(1.0)
    theta = theta.at[0, 2].set(-1.0)
    theta = theta.at[0, 6].set(-1.0 / 3.0)
    theta = theta.at[1, 0].set(_A * _B)
    theta = theta.at[1, 1].set(_A)
    theta = theta.at[1, 2].set(-_A * _C)
    return theta


def adjust_rhs(x: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    x_clipped = jnp.clip(x, -100.0, 100.0)
    rhs_clipped = jnp.clip(rhs, -1000.0, 1000.0)
    return jnp.where(jnp.isfinite(x_clipped).all(), rhs_clipped, jnp.zeros_like(rhs_clipped))


def make_state_evolution(drift_fn: Callable, diffusion_coeff) -> ContinuousTimeStateEvolution:
    return ContinuousTimeStateEvolution(
        drift=lambda x, u, t: adjust_rhs(x, drift_fn(x, u, t)),
        diffusion_coefficient=lambda x, u, t: diffusion_coeff * jnp.eye(STATE_DIM),
    )


H_OBS = jnp.eye(OBS_DIM, STATE_DIM)
INITIAL_CONDITION_KWARGS = dict(
    initial_condition=dist.MultivariateNormal(
        loc=jnp.zeros(STATE_DIM),
        covariance_matrix=jnp.eye(STATE_DIM),
    ),
)


def model_with_true_drift(obs_times=None, obs_values=None):
    return dsx.sample(
        "f",
        DynamicalModel(
            state_evolution=make_state_evolution(
                lambda x, u, t: fitzhugh_nagumo_drift(x),
                diffusion_coeff=SIGMA_X_TRUE,
            ),
            observation_model=LinearGaussianObservation(
                H=H_OBS,
                R=SIGMA_Y_TRUE**2 * jnp.eye(OBS_DIM),
            ),
            **INITIAL_CONDITION_KWARGS,
        ),
        obs_times=obs_times,
        obs_values=obs_values,
    )


def build_discovery_model(prior_name: str):
    prior_name = prior_name.lower()

    def model(obs_times=None, obs_values=None):
        if prior_name == "laplace":
            theta = numpyro.sample(
                "Theta",
                dist.Laplace(0.0, 0.1).expand([STATE_DIM, N_TERMS]).to_event(2),
            )
        elif prior_name == "horseshoe":
            tau = numpyro.sample("tau", dist.HalfCauchy(0.1))
            lam = numpyro.sample(
                "lambda_local",
                dist.HalfCauchy(1.0).expand([STATE_DIM, N_TERMS]).to_event(2),
            )
            scale = tau * lam + 1e-4
            theta = numpyro.sample(
                "Theta",
                dist.Normal(0.0, scale).to_event(2),
            )
        else:
            raise ValueError(f"Unsupported prior_name: {prior_name}")

        sigma_x = numpyro.sample("sigma_x", dist.HalfNormal(0.1))
        sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.5))

        def drift(x, u, t):
            return theta @ monomials(x)

        return dsx.sample(
            "f",
            DynamicalModel(
                state_evolution=make_state_evolution(drift, diffusion_coeff=sigma_x),
                observation_model=LinearGaussianObservation(
                    H=H_OBS,
                    R=sigma_y**2 * jnp.eye(OBS_DIM),
                ),
                **INITIAL_CONDITION_KWARGS,
            ),
            obs_times=obs_times,
            obs_values=obs_values,
        )

    def conditioned_model(obs_times=None, obs_values=None):
        with Filter():
            return model(obs_times=obs_times, obs_values=obs_values)

    return conditioned_model


@dataclass
class RunResult:
    prior_name: str
    method: str
    point_params: dict
    metrics: dict


def _gaussian_quantiles(mu: jnp.ndarray, sigma: jnp.ndarray, alpha: float = 0.1):
    z = 1.6448536269514722 if alpha == 0.1 else 1.959963984540054
    lo = mu - z * sigma
    hi = mu + z * sigma
    return lo, hi


def _posterior_theta_stats(theta_samples: jnp.ndarray) -> dict:
    theta_mean = jnp.mean(theta_samples, axis=0)
    theta_std = jnp.std(theta_samples, axis=0)
    lo, hi = _gaussian_quantiles(theta_mean, theta_std, alpha=0.1)
    return {
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_ci90_lo": lo,
        "theta_ci90_hi": hi,
    }


def support_metrics_bayesian(
    theta_true: jnp.ndarray,
    theta_samples: jnp.ndarray,
    coeff_threshold: float,
    inclusion_prob_threshold: float,
) -> dict:
    true_nz = jnp.abs(theta_true) > EPS
    inclusion_prob = jnp.mean(jnp.abs(theta_samples) >= coeff_threshold, axis=0)
    pred_nz = inclusion_prob >= inclusion_prob_threshold

    tp = int(jnp.logical_and(true_nz, pred_nz).sum())
    fp = int(jnp.logical_and(~true_nz, pred_nz).sum())
    fn = int(jnp.logical_and(true_nz, ~pred_nz).sum())

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_inclusion_prob": float(jnp.mean(inclusion_prob)),
    }


def drift_field_rmse(theta_hat: jnp.ndarray) -> float:
    grid_v = jnp.linspace(-2.0, 2.0, 41)
    grid_w = jnp.linspace(-1.5, 2.5, 41)
    vv, ww = jnp.meshgrid(grid_v, grid_w, indexing="ij")
    points = jnp.stack([vv.ravel(), ww.ravel()], axis=-1)
    true_drift = jnp.stack([fitzhugh_nagumo_drift(x) for x in points], axis=0)
    hat_drift = jnp.stack([theta_hat @ monomials(x) for x in points], axis=0)
    return float(jnp.sqrt(jnp.mean((hat_drift - true_drift) ** 2)))


def _posterior_filter_metrics(
    model,
    posterior_samples: dict,
    filter_key: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    true_states: jnp.ndarray,
    num_eval_samples: int,
):
    n_avail = posterior_samples["Theta"].shape[0]
    n_use = min(num_eval_samples, n_avail)

    sub_samples = {k: v[:n_use] for k, v in posterior_samples.items()}
    pred = Predictive(model, posterior_samples=sub_samples, exclude_deterministic=False)(
        filter_key,
        obs_times=obs_times,
        obs_values=obs_values,
    )

    fm = pred["f_filtered_states_mean"]  # (S, T[, +1], 2)
    fc = pred["f_filtered_states_cov_diag"]  # (S, T[, +1], 2)

    t_len = min(fm.shape[1], true_states.shape[0])
    fm = fm[:, :t_len, :]
    fc = fc[:, :t_len, :]
    st = true_states[:t_len]

    # total posterior predictive variance = E[var|theta] + var(E[x|theta])
    mean_of_means = jnp.mean(fm, axis=0)
    var_of_means = jnp.var(fm, axis=0)
    mean_of_vars = jnp.mean(fc, axis=0)
    total_var = jnp.maximum(var_of_means + mean_of_vars, 1e-10)
    total_std = jnp.sqrt(total_var)

    err = mean_of_means - st
    rmse = float(jnp.sqrt(jnp.mean(err**2)))
    mae = float(jnp.mean(jnp.abs(err)))
    coverage_1sigma = float(jnp.mean(jnp.abs(err) <= total_std))
    nll = float(
        jnp.mean(0.5 * (jnp.log(2 * jnp.pi * total_var) + (err**2) / total_var))
    )

    return {
        "state_rmse": rmse,
        "state_mae": mae,
        "state_coverage_1sigma": coverage_1sigma,
        "state_diag_gaussian_nll": nll,
    }


def _credible_interval_coverage(samples: jnp.ndarray, truth: jnp.ndarray, alpha: float = 0.1) -> float:
    lo = jnp.quantile(samples, alpha / 2.0, axis=0)
    hi = jnp.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return float(jnp.mean(jnp.logical_and(truth >= lo, truth <= hi)))


def evaluate_posterior(
    model,
    posterior_samples: dict,
    theta_true: jnp.ndarray,
    true_states: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    filter_key: jnp.ndarray,
    coeff_threshold: float,
    inclusion_prob_threshold: float,
    filter_eval_samples: int,
) -> tuple[dict, dict]:
    theta_samples = posterior_samples["Theta"]
    sigma_x_samples = posterior_samples["sigma_x"]
    sigma_y_samples = posterior_samples["sigma_y"]

    theta_mean = jnp.mean(theta_samples, axis=0)
    sigma_x_mean = float(jnp.mean(sigma_x_samples))
    sigma_y_mean = float(jnp.mean(sigma_y_samples))

    coeff_mae = float(jnp.mean(jnp.abs(theta_mean - theta_true)))
    coeff_rmse = float(jnp.sqrt(jnp.mean((theta_mean - theta_true) ** 2)))

    support = support_metrics_bayesian(
        theta_true,
        theta_samples,
        coeff_threshold=coeff_threshold,
        inclusion_prob_threshold=inclusion_prob_threshold,
    )

    filt = _posterior_filter_metrics(
        model=model,
        posterior_samples=posterior_samples,
        filter_key=filter_key,
        obs_times=obs_times,
        obs_values=obs_values,
        true_states=true_states,
        num_eval_samples=filter_eval_samples,
    )

    theta_ci90_coverage = _credible_interval_coverage(theta_samples, theta_true, alpha=0.1)
    sigma_x_ci90_coverage = float(
        jnp.quantile(sigma_x_samples, 0.05) <= SIGMA_X_TRUE <= jnp.quantile(sigma_x_samples, 0.95)
    )
    sigma_y_ci90_coverage = float(
        jnp.quantile(sigma_y_samples, 0.05) <= SIGMA_Y_TRUE <= jnp.quantile(sigma_y_samples, 0.95)
    )

    metrics = {
        "coeff_mae": coeff_mae,
        "coeff_rmse": coeff_rmse,
        "drift_field_rmse": drift_field_rmse(theta_mean),
        "sigma_x_mean": sigma_x_mean,
        "sigma_y_mean": sigma_y_mean,
        "sigma_x_rel_err_pct": abs(sigma_x_mean - SIGMA_X_TRUE) / SIGMA_X_TRUE * 100.0,
        "sigma_y_rel_err_pct": abs(sigma_y_mean - SIGMA_Y_TRUE) / SIGMA_Y_TRUE * 100.0,
        "theta_ci90_coverage": theta_ci90_coverage,
        "sigma_x_ci90_coverage": sigma_x_ci90_coverage,
        "sigma_y_ci90_coverage": sigma_y_ci90_coverage,
    }
    metrics.update(support)
    metrics.update(filt)

    point_params = {
        "Theta": theta_mean,
        "sigma_x": sigma_x_mean,
        "sigma_y": sigma_y_mean,
    }
    return point_params, metrics


def run_map(
    model,
    key: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    true_states: jnp.ndarray,
    theta_true: jnp.ndarray,
    num_steps: int,
    step_size: float,
    support_threshold: float,
    inclusion_prob_threshold: float,
    filter_eval_samples: int,
    filter_key: jnp.ndarray,
):
    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=step_size), loss=Trace_ELBO())
    svi_result = svi.run(key, num_steps=num_steps, obs_times=obs_times, obs_values=obs_values)
    map_params = guide.median(svi_result.params)

    # Treat MAP as degenerate posterior for unified Bayesian metrics.
    posterior_samples = {
        "Theta": jnp.expand_dims(map_params["Theta"], axis=0),
        "sigma_x": jnp.expand_dims(map_params["sigma_x"], axis=0),
        "sigma_y": jnp.expand_dims(map_params["sigma_y"], axis=0),
    }
    point_params, metrics = evaluate_posterior(
        model=model,
        posterior_samples=posterior_samples,
        theta_true=theta_true,
        true_states=true_states,
        obs_times=obs_times,
        obs_values=obs_values,
        filter_key=filter_key,
        coeff_threshold=support_threshold,
        inclusion_prob_threshold=inclusion_prob_threshold,
        filter_eval_samples=1,
    )
    return point_params, metrics, svi_result.losses


def run_autonormal_svi(
    model,
    key: jnp.ndarray,
    posterior_key: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    true_states: jnp.ndarray,
    theta_true: jnp.ndarray,
    num_steps: int,
    step_size: float,
    posterior_samples_n: int,
    support_threshold: float,
    inclusion_prob_threshold: float,
    filter_eval_samples: int,
    filter_key: jnp.ndarray,
):
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=step_size), loss=Trace_ELBO())
    svi_result = svi.run(key, num_steps=num_steps, obs_times=obs_times, obs_values=obs_values)

    posterior_samples = guide.sample_posterior(
        posterior_key,
        svi_result.params,
        sample_shape=(posterior_samples_n,),
    )
    point_params, metrics = evaluate_posterior(
        model=model,
        posterior_samples=posterior_samples,
        theta_true=theta_true,
        true_states=true_states,
        obs_times=obs_times,
        obs_values=obs_values,
        filter_key=filter_key,
        coeff_threshold=support_threshold,
        inclusion_prob_threshold=inclusion_prob_threshold,
        filter_eval_samples=filter_eval_samples,
    )
    return point_params, metrics, svi_result.losses


def run_mcmc(
    model,
    key: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    true_states: jnp.ndarray,
    theta_true: jnp.ndarray,
    warmup: int,
    samples: int,
    chains: int,
    support_threshold: float,
    inclusion_prob_threshold: float,
    filter_eval_samples: int,
    filter_key: jnp.ndarray,
):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=True,
    )
    mcmc.run(key, obs_times=obs_times, obs_values=obs_values)
    posterior_samples = mcmc.get_samples(group_by_chain=False)

    point_params, metrics = evaluate_posterior(
        model=model,
        posterior_samples=posterior_samples,
        theta_true=theta_true,
        true_states=true_states,
        obs_times=obs_times,
        obs_values=obs_values,
        filter_key=filter_key,
        coeff_threshold=support_threshold,
        inclusion_prob_threshold=inclusion_prob_threshold,
        filter_eval_samples=filter_eval_samples,
    )
    return point_params, metrics, None


def print_results(theta_true: jnp.ndarray, results: list[RunResult]):
    print("\n=== Ground-Truth Theta ===")
    print(jnp.round(theta_true, 4))

    print("\n=== Inferred Theta Means (by prior/method) ===")
    for result in results:
        print(f"\n[{result.prior_name} | {result.method}]")
        print(jnp.round(result.point_params["Theta"], 4))

    columns = [
        "coeff_mae",
        "coeff_rmse",
        "drift_field_rmse",
        "precision",
        "recall",
        "f1",
        "mean_inclusion_prob",
        "sigma_x_mean",
        "sigma_y_mean",
        "sigma_x_rel_err_pct",
        "sigma_y_rel_err_pct",
        "theta_ci90_coverage",
        "sigma_x_ci90_coverage",
        "sigma_y_ci90_coverage",
        "state_rmse",
        "state_mae",
        "state_coverage_1sigma",
        "state_diag_gaussian_nll",
    ]
    print("\n=== Formal Evaluation (posterior-aware) ===")
    header = "prior/method".ljust(22) + " ".join([c.rjust(20) for c in columns])
    print(header)
    print("-" * len(header))
    for result in results:
        row = [f"{result.prior_name}/{result.method}".ljust(22)]
        for col in columns:
            val = result.metrics.get(col, float("nan"))
            row.append(f"{val:20.6f}")
        print(" ".join(row))


def _parse_csv_arg(arg: str, allowed: set[str]) -> list[str]:
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty selection provided.")
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unsupported option '{p}'. Allowed: {sorted(allowed)}")
    return parts


def main():
    args = parse_args()
    methods = _parse_csv_arg(args.methods, {"map", "autonormal", "mcmc"})
    priors = _parse_csv_arg(args.priors, {"laplace", "horseshoe"})

    obs_times = jnp.arange(0.0, args.tmax, args.dt)
    theta_gt = true_theta()

    key = jr.PRNGKey(args.seed)
    key, k_data = jr.split(key)

    predictive = Predictive(model_with_true_drift, num_samples=1, exclude_deterministic=False)
    with SDESimulator():
        synthetic = predictive(k_data, obs_times=obs_times)
    obs_values = synthetic["observations"][0]
    true_states = synthetic["states"][0]

    results: list[RunResult] = []

    for prior_name in priors:
        model = build_discovery_model(prior_name)

        if "map" in methods:
            key, k_fit, k_filter = jr.split(key, 3)
            point_params, metrics, _ = run_map(
                model=model,
                key=k_fit,
                obs_times=obs_times,
                obs_values=obs_values,
                true_states=true_states,
                theta_true=theta_gt,
                num_steps=args.num_steps,
                step_size=args.step_size,
                support_threshold=args.support_threshold,
                inclusion_prob_threshold=args.inclusion_prob_threshold,
                filter_eval_samples=1,
                filter_key=k_filter,
            )
            results.append(
                RunResult(
                    prior_name=prior_name,
                    method="map",
                    point_params=point_params,
                    metrics=metrics,
                )
            )

        if "autonormal" in methods:
            key, k_fit, k_post, k_filter = jr.split(key, 4)
            point_params, metrics, _ = run_autonormal_svi(
                model=model,
                key=k_fit,
                posterior_key=k_post,
                obs_times=obs_times,
                obs_values=obs_values,
                true_states=true_states,
                theta_true=theta_gt,
                num_steps=args.num_steps,
                step_size=args.step_size,
                posterior_samples_n=args.posterior_samples,
                support_threshold=args.support_threshold,
                inclusion_prob_threshold=args.inclusion_prob_threshold,
                filter_eval_samples=args.filter_eval_samples,
                filter_key=k_filter,
            )
            results.append(
                RunResult(
                    prior_name=prior_name,
                    method="autonormal",
                    point_params=point_params,
                    metrics=metrics,
                )
            )

        if "mcmc" in methods:
            key, k_fit, k_filter = jr.split(key, 3)
            point_params, metrics, _ = run_mcmc(
                model=model,
                key=k_fit,
                obs_times=obs_times,
                obs_values=obs_values,
                true_states=true_states,
                theta_true=theta_gt,
                warmup=args.mcmc_warmup,
                samples=args.mcmc_samples,
                chains=args.mcmc_chains,
                support_threshold=args.support_threshold,
                inclusion_prob_threshold=args.inclusion_prob_threshold,
                filter_eval_samples=args.filter_eval_samples,
                filter_key=k_filter,
            )
            results.append(
                RunResult(
                    prior_name=prior_name,
                    method="mcmc",
                    point_params=point_params,
                    metrics=metrics,
                )
            )

    print_results(theta_gt, results)


if __name__ == "__main__":
    # MCMC can trigger many compile warnings; keep default warnings behavior unchanged.
    main()
