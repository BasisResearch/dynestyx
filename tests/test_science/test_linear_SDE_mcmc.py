import jax.numpy as jnp
import jax.random as jr

import arviz as az
from numpyro.infer import MCMC, NUTS
import pytest

from tests.fixtures import data_conditioned_continuous_time_linear_sde  # noqa: F401
from tests.test_utils import get_output_dir


SAVE_FIG = True


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(data_conditioned_continuous_time_linear_sde, num_samples):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        use_controls,
    ) = data_conditioned_continuous_time_linear_sde

    # Set output dir based on whether controls are used
    output_dir_name = "test_linear_SDE_mcmc" + ("_controlled" if use_controls else "")
    OUTPUT_DIR = get_output_dir(output_dir_name)

    obs_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        states = synthetic["states"].squeeze(0)  # shape: (T, 2)
        observations = synthetic["observations"].squeeze(0)  # shape: (T, 2) - fully observed

        # Plot states and observations for both components
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Component 0
        axes[0].plot(
            obs_times.squeeze(0),
            states[:, 0],
            label="x[0] (state)",
            linewidth=2,
        )
        axes[0].plot(
            obs_times.squeeze(0),
            observations[:, 0],
            label="y[0] (observation)",
            linestyle="--",
            alpha=0.7,
        )
        axes[0].set_ylabel("Component 0")
        axes[0].set_title("Fully Observed Linear SDE - Component 0")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Component 1
        axes[1].plot(
            obs_times.squeeze(0),
            states[:, 1],
            label="x[1] (state)",
            linewidth=2,
        )
        axes[1].plot(
            obs_times.squeeze(0),
            observations[:, 1],
            label="y[1] (observation)",
            linestyle="--",
            alpha=0.7,
        )
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Component 1")
        axes[1].set_title("Fully Observed Linear SDE - Component 1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    mcmc_key = jr.PRNGKey(0)
    nuts_kernel = NUTS(data_conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_samples)
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    # Check that all parameters are present
    assert "A" in posterior_samples
    assert "B" in posterior_samples
    assert "L" in posterior_samples
    assert "sigma_observation" in posterior_samples

    posterior_A = posterior_samples["A"]  # shape: (num_samples, 2, 2)
    posterior_B = posterior_samples["B"]  # shape: (num_samples, 2, 1)
    posterior_L = posterior_samples["L"]  # shape: (num_samples, 2, 2)
    posterior_sigma_obs = posterior_samples["sigma_observation"]  # shape: (num_samples,)

    assert len(posterior_A) == num_samples
    assert posterior_A.shape == (num_samples, 2, 2)
    assert posterior_B.shape == (num_samples, 2, 1)
    assert posterior_L.shape == (num_samples, 2, 2)
    assert not jnp.isnan(posterior_A).any()
    assert not jnp.isnan(posterior_B).any()
    assert not jnp.isnan(posterior_L).any()
    assert not jnp.isnan(posterior_sigma_obs).any()
    assert not jnp.isinf(posterior_A).any()
    assert not jnp.isinf(posterior_B).any()
    assert not jnp.isinf(posterior_L).any()
    assert not jnp.isinf(posterior_sigma_obs).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot posteriors for A matrix entries
        true_A = true_params["A"]
        az.plot_posterior(
            posterior_A[:, 0, 0],
            hdi_prob=0.95,
            ref_val=true_A[0, 0].item(),
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("A[0,0]")
        
        az.plot_posterior(
            posterior_A[:, 0, 1],
            hdi_prob=0.95,
            ref_val=true_A[0, 1].item(),
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("A[0,1]")
        
        az.plot_posterior(
            posterior_A[:, 1, 0],
            hdi_prob=0.95,
            ref_val=true_A[1, 0].item(),
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("A[1,0]")
        
        az.plot_posterior(
            posterior_A[:, 1, 1],
            hdi_prob=0.95,
            ref_val=true_A[1, 1].item(),
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("A[1,1]")
        
        az.plot_posterior(
            posterior_sigma_obs,
            hdi_prob=0.95,
            ref_val=true_params["sigma_observation"].item(),
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("sigma_observation")
        
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "posteriors.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Check that posteriors are reasonable (true values should be in credible intervals)
    # Check A matrix entries
    true_A = true_params["A"]
    for i in range(2):
        for j in range(2):
            posterior_A_ij = posterior_A[:, i, j]
            true_val = true_A[i, j].item()
            
            hdi_data = az.hdi(posterior_A_ij, hdi_prob=0.95)
            hdi_min = hdi_data["x"].sel(hdi="lower").item()
            hdi_max = hdi_data["x"].sel(hdi="higher").item()
            
            assert hdi_min <= true_val <= hdi_max, (
                f"True A[{i},{j}] {true_val} not in HDI [{hdi_min}, {hdi_max}]"
            )
    
    # Check sigma_observation
    hdi_data = az.hdi(posterior_sigma_obs, hdi_prob=0.95)
    hdi_min = hdi_data["x"].sel(hdi="lower").item()
    hdi_max = hdi_data["x"].sel(hdi="higher").item()
    true_sigma_obs = true_params["sigma_observation"].item()
    assert hdi_min <= true_sigma_obs <= hdi_max, (
        f"True sigma_observation {true_sigma_obs} not in HDI [{hdi_min}, {hdi_max}]"
    )