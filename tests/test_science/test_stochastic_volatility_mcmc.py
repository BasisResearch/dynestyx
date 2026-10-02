import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS

from tests.arviz_utils import hdi_bounds, save_posterior_plot
from tests.fixtures import data_conditioned_stochastic_volatility  # noqa: F401
from tests.test_utils import get_output_dir

SAVE_FIG = True


@pytest.mark.parametrize("num_samples", [250])
def test_mcmc_inference(data_conditioned_stochastic_volatility, num_samples):  # noqa: F811
    (
        data_conditioned_model,
        true_params,
        synthetic,
        _use_controls,
    ) = data_conditioned_stochastic_volatility

    output_dir_name = "test_stochastic_volatility_mcmc"
    OUTPUT_DIR = get_output_dir(output_dir_name)

    obs_times = synthetic["times"]

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        states_1d = jnp.squeeze(synthetic["states"])
        obs_1d = jnp.squeeze(synthetic["observations"])
        times_1d = jnp.squeeze(obs_times)
        plt.plot(times_1d, states_1d, label="log-variance")
        plt.plot(times_1d, obs_1d, label="observations", linestyle="--")
        plt.legend()
        plt.savefig(OUTPUT_DIR / "data_generation.png", dpi=150, bbox_inches="tight")
        plt.close()

    mcmc_key = jr.PRNGKey(0)
    nuts_kernel = NUTS(data_conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_samples)
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()

    assert "phi" in posterior_samples
    posterior_phi = posterior_samples["phi"]
    assert len(posterior_phi) == num_samples
    assert not jnp.isnan(posterior_phi).any()
    assert not jnp.isinf(posterior_phi).any()

    if SAVE_FIG and OUTPUT_DIR is not None:
        import matplotlib.pyplot as plt

        save_posterior_plot(
            posterior_phi,
            name="phi",
            output_path=OUTPUT_DIR / "posterior_phi.png",
            ref_val=true_params["phi"].item(),
        )

    assert jnp.abs(posterior_phi.mean() - true_params["phi"]) < 0.3

    hdi_min, hdi_max = hdi_bounds(posterior_phi)
    assert hdi_min <= true_params["phi"] <= hdi_max, (
        f"True phi {true_params['phi']} not in HDI {hdi_min}, {hdi_max}"
    )
