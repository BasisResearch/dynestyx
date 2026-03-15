from dynestyx import DynamicalModel, ContinuousTimeStateEvolution, LinearGaussianObservation
from dynestyx import Filter, Simulator
from dynestyx.inference.filter_configs import ContinuousTimeDPFConfig, ContinuousTimeEKFConfig
from dynestyx import LTI_continuous
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import dynestyx as dsx
import jax

def model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    """Continuous-time LTI using LTI_continuous factory: only rho = A[1,0] is sampled."""
    rho = numpyro.sample("rho", dist.Uniform(0.0, 5.0))
    state_dim = 2
    A = jnp.array([[-1.0, 0.0], [rho, -1.0]])
    L = jnp.eye(state_dim)
    H = jnp.array([[0.0, 1.0]])
    R = jnp.array([[1.0**2]])
    B = jnp.array([[0.0], [10.0]])
    dynamics = LTI_continuous(A=A, L=L, H=H, R=R, B=B)
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )

from numpyro.infer import Predictive

from dynestyx.simulators import SDESimulator

with SDESimulator():
    predictive_model = Predictive(model, num_samples=1, exclude_deterministic=False, params={"rho": 0.3})
    sim_data = predictive_model(jax.random.PRNGKey(0), predict_times=jnp.arange(0.0, 10.0, 0.05))

print("Simulated data", sim_data)

with numpyro.handlers.seed(rng_seed=0):
    with Filter(filter_config=ContinuousTimeEKFConfig()):
        result = model(obs_times=jnp.arange(0.0, 10.0, 0.05), obs_values=jnp.ones((200, 1)), predict_times=jnp.arange(0.0, 10.0, 0.05))

print("Filter didn't crash")

obs_times = jnp.linspace(start=0.0, stop=1.0, num=20)
obs_values = jnp.ones((20, 1))
predict_times = jnp.linspace(start=1.0, stop=2.0, num=20)

with SDESimulator():
    with Filter(
        filter_config=ContinuousTimeEKFConfig(record_filtered_states_mean=True)
    ):
        predictive = Predictive(
            model,
            num_samples=1,
            exclude_deterministic=False,
        )
        result = predictive(
            jax.random.PRNGKey(0),
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )

print("Simulator within filter didn't crash")

# Plot filtered states/means and rollout predictions
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Filtered states at obs_times
filtered_mean = np.asarray(result["f_filtered_states_mean"]).squeeze(0)
for i in range(filtered_mean.shape[1]):
    axes[0].plot(
        np.asarray(obs_times),
        filtered_mean[:, i],
        "o-",
        label=f"Filtered $x_{i}$",
        markersize=4,
    )

# Predicted states at predict_times
pred_states = np.asarray(result["f_predicted_states"]).squeeze(0)
pred_times = np.asarray(result["f_predicted_times"]).squeeze(0)
for i in range(pred_states.shape[1]):
    axes[0].plot(
        pred_times,
        pred_states[:, i],
        "-",
        label=f"Predicted $x_{i}$",
        alpha=0.8,
    )

axes[0].axvline(obs_times[-1], color="gray", linestyle="--", alpha=0.7, label="Last obs")
axes[0].set_ylabel("State")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Filtered means vs rollout predictions")

# Observations
pred_obs = np.asarray(result["f_predicted_observations"]).squeeze(0)
axes[1].plot(np.asarray(obs_times), np.asarray(obs_values).squeeze(-1), "o-", label="Obs")
axes[1].plot(pred_times, pred_obs, "-", label="Predicted obs", alpha=0.8)
axes[1].axvline(obs_times[-1], color="gray", linestyle="--", alpha=0.7)
axes[1].set_ylabel("Observation")
axes[1].set_xlabel("Time")
axes[1].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("test_predict_times_plot.png", dpi=150)
print("Saved test_predict_times_plot.png")
plt.close()