"""Distribution implementations used by dynestyx models."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro.distributions import constraints


class MixedStateDistribution(dist.Distribution):
    """Joint distribution for an SLDS mixed state `[z, x...]`.

    The first event entry is a discrete categorical state encoded as a scalar;
    the remaining event entries are continuous and conditionally Gaussian given
    the categorical state.

    Sampling and scoring delegate to NumPyro's `Categorical` and
    `MultivariateNormal` distributions:

    ```
    z ~ Categorical(categorical_probs)
    x | z ~ MultivariateNormal(continuous_locs[z], continuous_covs[z])
    ```
    """

    arg_constraints = {}
    support = constraints.real_vector
    pytree_data_fields = ("categorical_probs", "continuous_locs", "continuous_covs")
    pytree_aux_fields = ("num_categories", "continuous_state_dim")

    def __init__(
        self,
        categorical_probs,
        continuous_locs,
        continuous_covs,
        validate_args=None,
    ):
        self.categorical_probs = categorical_probs
        self.continuous_locs = continuous_locs
        self.continuous_covs = continuous_covs
        self.num_categories = int(categorical_probs.shape[-1])
        self.continuous_state_dim = int(continuous_locs.shape[-1])
        super().__init__(
            batch_shape=(),
            event_shape=(self.continuous_state_dim + 1,),
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        key_z, key_x = jr.split(key)
        z = dist.Categorical(probs=self.categorical_probs).sample(key_z, sample_shape)
        means = self.continuous_locs[z]
        covs = self.continuous_covs[z]
        x = dist.MultivariateNormal(means, covariance_matrix=covs).sample(key_x)
        return jnp.concatenate([z[..., None].astype(x.dtype), x], axis=-1)

    def log_prob(self, value):
        z = jnp.rint(value[..., 0]).astype(jnp.int32)
        x = value[..., 1:]
        return dist.Categorical(probs=self.categorical_probs).log_prob(
            z
        ) + dist.MultivariateNormal(
            self.continuous_locs[z], covariance_matrix=self.continuous_covs[z]
        ).log_prob(x)


__all__ = ["MixedStateDistribution"]
