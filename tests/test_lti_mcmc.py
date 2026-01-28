# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS, Predictive
# from numpyro.handlers import seed, trace
# import jax.numpy as jnp
# from jax import random
# from effectful.ops.semantics import handler, fwd
# from dsx.handlers import (
#     BaseSolver,
#     States,
#     BaseCDDynamaxLogFactorAdder,
#     Condition,
#     KalmanSolver,
# )
# from dsx.dynamical_models import ContinuousTimeDynamicalModel
# import dsx
# from dsx.ops import Trajectory
# import jax
# from jax import lax
# from jax.scipy.linalg import expm

# global_key = random.PRNGKey(0)
# PLOT = True


# def get_key():
#     global global_key
#     key, global_key = random.split(global_key)
#     return key


# jax.config.update("jax_enable_x64", True)


# class GaussianLTIDynamics(ContinuousTimeDynamicalModel):
#     """Dynamical model that takes parameters passed to a function."""

#     def __init__(self, x_0, t_0, params: dict, state_name: str = "x"):
#         self.params = params
#         self.state_name = state_name
#         self.x_0 = x_0
#         self.t_0 = t_0


# class GaussianLTISolver(BaseSolver):
#     def solve(self, times, dynamics: GaussianLTIDynamics) -> States:
#         A = dynamics.params["A"]
#         L = dynamics.params["L"]
#         x0 = dynamics.x_0
#         ts = times

#         A = jnp.asarray(A, dtype=jnp.float64)
#         if A.ndim == 0:
#             A = A[None, None]  # (1,1)
#         n = A.shape[0]

#         L = jnp.asarray(L, dtype=jnp.float64)
#         if L.ndim == 0:
#             L = L[None, None]

#         x0 = jnp.asarray(x0, dtype=jnp.float64)
#         if x0.ndim == 0:
#             x0 = x0[None]  # (n,)

#         ts = jnp.asarray(ts, dtype=jnp.float64)
#         dts = ts[1:] - ts[:-1]

#         Qc = L @ L.T

#         zero = jnp.zeros_like(A)
#         base_M = jnp.block([[A, Qc], [zero, -A.T]])
#         eye_n = jnp.eye(n, dtype=jnp.float64)

#         key = get_key()

#         def step(carry, dt):
#             x, key = carry
#             key, subkey = random.split(key)

#             expM = expm(base_M * dt)
#             Ad = expM[:n, :n]
#             S = expM[:n, n:]
#             Qd = S @ Ad.T
#             Qd = 0.5 * (Qd + Qd.T)

#             # If there is effectively no process noise, skip sampling
#             def no_noise(_):
#                 return jnp.zeros((n,), dtype=jnp.float64)

#             def with_noise(k):
#                 Ld = jnp.linalg.cholesky(Qd + 1e-9 * eye_n)
#                 return Ld @ random.normal(k, (n,))

#             noise = lax.cond(
#                 jnp.allclose(Qc, jnp.zeros_like(Qc)), no_noise, with_noise, subkey
#             )
#             x_next = Ad @ x + noise
#             return (x_next, key), x_next

#         def scan_body(init_x):
#             (x_last, _), xs_rest = lax.scan(step, (init_x, key), dts)
#             xs = jnp.vstack([init_x, xs_rest])
#             return xs

#         xs = scan_body(x0)

#         # Return shape:
#         # - if n == 1, squeeze to (T,)
#         # - else, return (T, n)
#         if n == 1:
#             xs_out = xs[:, 0]
#         else:
#             xs_out = xs

#         return {dynamics.state_name: xs_out}


# class GaussianEmissionLogFactorAdder(BaseCDDynamaxLogFactorAdder):
#     def add_log_factors(
#         self, name: str, dynamics: GaussianLTIDynamics, obs: Trajectory
#     ):
#         if not isinstance(dynamics, GaussianLTIDynamics):
#             raise NotImplementedError(
#                 f"GaussianEmissionLogFactorAdder only works with LTIDynamics, got {type(dynamics)}"
#             )
#         trajectory_fn = fwd()
#         obs_times, obs_states = obs

#         pred_states = trajectory_fn(obs_times)

#         mll_total = 0.0
#         for state_name in obs_states:
#             mll = (
#                 dist.MultivariateNormal(pred_states[state_name], dynamics.params["R"])
#                 .log_prob(obs_states[state_name])
#                 .sum()
#             )
#             mll_total += mll

#         numpyro.factor(f"{name}_gaussian_emission_log_factor", mll_total)


# def model():
#     """Model that samples phase and uses it in dynamics."""
#     # A = numpyro.sample("A", dist.Normal(0.0, 1.0).expand((2, 2)))
#     A_0_0 = numpyro.sample("A_0_0", dist.Normal(0.0, 1.0))
#     A_0_1 = numpyro.sample("A_0_1", dist.Normal(0.0, 0.2))
#     A_1_0 = numpyro.sample("A_1_0", dist.Normal(0.0, 0.2))
#     A_1_1 = numpyro.sample("A_1_1", dist.Normal(0.0, 1.0))
#     A = numpyro.deterministic("A", jnp.array([[A_0_0, A_0_1], [A_1_0, A_1_1]]))
#     R = jnp.eye(2)
#     L = jnp.eye(2)
#     dynamics = GaussianLTIDynamics(
#         x_0=jnp.array([0.0, 0.0]),
#         t_0=0.0,
#         params={"A": A, "R": R, "L": L},
#         state_name="x",
#     )
#     return dsx.sample_ds("f", dynamics, None)


# def conditioned_model(obs_data: dict[str, Trajectory]):
#     """Create a conditioned model with solver, log factor adder, and condition handlers."""

#     # TODO instantiating these out here and reusing in run_model broke. Jack why
#     # solver_handler = handler(DumbSolver())
#     # log_factor_adder_handler = handler(MSELogFactorAdder())
#     # condition_handler = handler(Condition(obs_data))

#     def run_model():
#         solver_handler = handler(KalmanSolver())
#         log_factor_adder_handler = handler(GaussianEmissionLogFactorAdder())
#         condition_handler = handler(Condition(obs_data))

#         with solver_handler:
#             with log_factor_adder_handler:
#                 with condition_handler:
#                     return model()

#     return run_model


# def test_forward_sampling_smoke():
#     """Smoke test of forward sampling."""
#     solver = handler(GaussianLTISolver())
#     seeded_model = seed(model, rng_seed=0)

#     # Trace the model to verify phase is sampled correctly
#     with trace() as trace_dict:
#         with solver:
#             trajectory_fn = seeded_model()

#     # Check that phase was sampled
#     assert "A" in trace_dict, "A should be sampled"
#     sampled_A = trace_dict["A"]["value"]

#     # Evaluate trajectory at some times
#     times = jnp.linspace(0.0, 10.0, 100)
#     result = trajectory_fn(times)

#     if PLOT:
#         import matplotlib.pyplot as plt

#         plt.plot(times, result["x"])
#         plt.show()
#     assert result["x"].shape[0] == times.shape[0]


# def run_mcmc_inference(
#     true_params: dict,
#     num_samples: int = 500,
#     num_warmup: int = 1_000,
#     rng_key: random.PRNGKey = get_key(),
# ):
#     """Run MCMC inference on synthetic data."""

#     def evaluated_model(times):
#         f = model()
#         return numpyro.deterministic("feval", f(times))

#     # Generate synthetic data using Predictive with ground truth phase
#     # Predictive handles seeding internally
#     solver = handler(GaussianLTISolver())

#     predictive = Predictive(evaluated_model, params=true_params, num_samples=1)

#     # Generate observations at some times
#     obs_times = jnp.linspace(0.0, 2.0, 100)

#     with solver:
#         samples = predictive(rng_key, obs_times)

#     obs_states = {"x": samples["feval"]["x"]}
#     obs_data = {"f": (obs_times, obs_states)}

#     # Create conditioned model
#     cond_model = conditioned_model(obs_data)

#     # Run NUTS MCMC
#     nuts_kernel = NUTS(cond_model)
#     mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
#     mcmc.run(rng_key)

#     # Get posterior samples
#     posterior_samples = mcmc.get_samples()

#     # Posterior predictive at evaluation times
#     eval_times = jnp.linspace(0.0, 2.0, 100)
#     predictive = Predictive(evaluated_model, posterior_samples)
#     with handler(GaussianLTISolver()):
#         pred_samples = predictive(rng_key, eval_times)

#     trajectory_evals = pred_samples["feval"]

#     return {
#         "true_A": true_params["A"],
#         "true_R": true_params["R"],
#         "true_L": true_params["L"],
#         "posterior_A": posterior_samples["A"],
#         "posterior_predictive": trajectory_evals["x"],
#         "eval_times": eval_times,
#         "obs_times": obs_times,
#         "obs_data": obs_data,
#     }


# def test_mcmc_smoke():
#     """Smoke test of MCMC inference."""
#     result = run_mcmc_inference(
#         true_params={"A": 1.0, "R": jnp.eye(1), "L": jnp.eye(1)},
#         num_samples=100,
#         num_warmup=50,
#     )
#     for q in ("A",):
#         assert f"posterior_{q}" in result
#         assert len(result[f"posterior_{q}"]) > 0


# if __name__ == "__main__":
#     # with jax.disable_jit():
#     test_forward_sampling_smoke()
#     # test_mcmc_smoke()

#     import matplotlib.pyplot as plt

#     true_params = {
#         "A": jnp.array([[1.0, -0.1], [0.0, 1.0]]),
#         "R": jnp.eye(2),
#         "L": jnp.eye(2) * 0.5,
#     }

#     result = run_mcmc_inference(
#         true_params=true_params,
#     )

#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     colors = ["b", "g", "r", "y"]
#     axes = axes.ravel()
#     for idx, (q, i, j) in enumerate(
#         (("A", 0, 0), ("A", 0, 1), ("A", 1, 0), ("A", 1, 1))
#     ):
#         ax = axes[idx]
#         ax.hist(
#             result[f"posterior_{q}"][:, i, j],
#             bins=50,
#             alpha=0.7,
#             label="Posterior",
#             color=colors[idx],
#         )
#         ax.axvline(
#             result[f"true_{q}"][i, j],
#             color=colors[idx],
#             linestyle="--",
#             linewidth=2,
#             label=f"True {q}[{i}, {j}]",
#         )
#         ax.set_xlabel(q)
#         ax.set_ylabel("Frequency")
#         ax.set_title(f"Posterior {q}[{i}, {j}]")
#         ax.legend()
#     plt.tight_layout()
#     plt.show()

#     # Plot confidence intervals for posterior predictive
#     pred_samples = result["posterior_predictive"]
#     eval_times = result["eval_times"]

#     # Compute percentiles
#     lower = jnp.percentile(pred_samples, 2.5, axis=0)
#     upper = jnp.percentile(pred_samples, 97.5, axis=0)
#     median = jnp.median(pred_samples, axis=0)

#     for i in range(2):
#         plt.fill_between(
#             eval_times,
#             lower[:, i],
#             upper[:, i],
#             alpha=0.3,
#             label="95% CI",
#             color=colors[i],
#         )
#         plt.plot(eval_times, median[:, i], colors[i], label=f"Median {i}")

#     # # Plot true trajectory
#     # true_traj = GaussianLTIDynamics(
#     #     x_0=jnp.array([0.0, 0.0]),
#     #     t_0=0.0,
#     #     params={"A": result["true_A"], "R": result["true_R"], "L": result["true_L"]},
#     #     state_name="x",
#     # ).solve(eval_times)["x"]
#     # plt.plot(eval_times, true_traj, "r--", label="True trajectory")

#     # Plot observations
#     for i in range(2):
#         obs_times, obs_states = result["obs_data"]["f"]
#         plt.scatter(
#             obs_times,
#             obs_states["x"][0, :, i],
#             color=colors[i],
#             s=5,
#             zorder=5,
#             label=f"Observations {i}",
#         )

#     plt.xlabel("Time")
#     plt.ylabel("x")
#     plt.title("Posterior Predictive Trajectory")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
