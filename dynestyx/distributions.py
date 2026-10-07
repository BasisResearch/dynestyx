"""Distribution implementations used by dynestyx models."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Int, Real
from numpyro.distributions import constraints


def _extract_and_validate_mixed_state(
    mixed_state: Real[Array, "*batch mixed_state_dim"],
    num_categories: int,
    *,
    rounding: bool = False,
) -> tuple[
    Int[Array, "*batch"],
    Real[Array, "*batch continuous_state_dim"],
    Bool[Array, "*batch"],
]:
    """Extract discrete and continuous components and check the discrete index.

    Returns a safe integer index, the unchanged continuous component, and a
    validity mask for the discrete component. Valid indices are finite integers
    in ``[0, num_categories)``; the continuous component is not validated here.
    Invalid indices are replaced with zero so indexing remains safe under JAX
    transformations. Callers must use the mask to reject invalid states or
    assign negative-infinite log probability. Exact equality is intentional:
    categorical samples packed into float states remain integer-valued.
    With ``rounding=True``, round to the nearest integer (ties to even) before
    validation; only the discrete component is rounded.
    """
    discrete_state = mixed_state[..., 0]
    if rounding:
        discrete_state = jnp.rint(discrete_state)
    valid = (
        jnp.isfinite(discrete_state)
        & (discrete_state == jnp.floor(discrete_state))
        & (discrete_state >= 0)
        & (discrete_state < num_categories)
    )
    index = jnp.where(valid, discrete_state, 0).astype(jnp.int32)
    return index, mixed_state[..., 1:], valid


class MixedStateDistribution(dist.Distribution):
    """Joint distribution for an SLDS mixed state `[z, x...]`.

    The first event entry is a discrete categorical state encoded as a scalar;
    the remaining event entries are continuous and conditionally Gaussian given
    the categorical state.

    ``rounding=False`` requires an integer-valued discrete coordinate in
    ``log_prob``. Set ``rounding=True`` to round it to the nearest integer
    (ties to even) before scoring; nonfinite or out-of-range results still
    receive negative-infinite log probability. Sampling is unchanged.

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
    pytree_aux_fields = ("num_categories", "continuous_state_dim", "rounding")
    rounding: bool

    def __init__(
        self,
        categorical_probs,
        continuous_locs,
        continuous_covs,
        validate_args=None,
        *,
        rounding: bool = False,
    ):
        self.rounding = rounding
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
        z, x, valid = _extract_and_validate_mixed_state(
            value, self.num_categories, rounding=self.rounding
        )
        log_prob = dist.Categorical(probs=self.categorical_probs).log_prob(
            z
        ) + dist.MultivariateNormal(
            self.continuous_locs[z], covariance_matrix=self.continuous_covs[z]
        ).log_prob(x)
        return jnp.where(valid, log_prob, -jnp.inf)


__all__ = ["MixedStateDistribution"]
