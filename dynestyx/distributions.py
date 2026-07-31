"""Probability distributions used by specialized dynestyx models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import lax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Real
from numpyro.distributions import constraints


class MixedStateDistribution(dist.Distribution):
    r"""Joint distribution for the discrete and continuous state of an SLDS.

    A switching linear dynamical system (SLDS) has a regime \(z\) and a
    continuous state \(x\):

    \[
    z \sim \operatorname{Categorical}(\pi), \qquad
    x \mid z \sim \mathcal{N}(\mu_z, \Sigma_z).
    \]

    Dynestyx represents the joint state as the homogeneous JAX vector
    ``[z, *x]``. The regime is therefore encoded in the first floating-point
    entry, although it is sampled and scored as a categorical integer.

    Leading batch dimensions are supported. The regime axis must be the final
    batch axis of ``continuous_locs`` and ``continuous_covariances``.

    Args:
        categorical_probs: Regime probabilities with shape
            ``(*batch, num_regimes)``.
        continuous_locs: Conditional means with shape
            ``(*batch, num_regimes, state_dim)``.
        continuous_covariances: Conditional covariance matrices with shape
            ``(*batch, num_regimes, state_dim, state_dim)``.
        validate_args: Whether NumPyro should validate samples and parameters.

    Attributes:
        categorical_probs: Regime probabilities.
        continuous_locs: Regime-conditional continuous-state means.
        continuous_covariances: Regime-conditional continuous-state
            covariances.
    """

    arg_constraints = {
        "categorical_probs": constraints.simplex,
        "continuous_locs": constraints.real,
        "continuous_covariances": constraints.positive_definite,
    }
    support = constraints.real_vector
    pytree_data_fields = (
        "categorical_probs",
        "continuous_locs",
        "continuous_covariances",
    )
    pytree_aux_fields = ("_batch_shape", "_event_shape")

    categorical_probs: Float[Array, "*batch num_regimes"]
    continuous_locs: Float[Array, "*batch num_regimes state_dim"]
    continuous_covariances: Float[Array, "*batch num_regimes state_dim state_dim"]

    def __init__(
        self,
        categorical_probs: Float[Array, "*batch num_regimes"],
        continuous_locs: Float[Array, "*batch num_regimes state_dim"],
        continuous_covariances: Float[Array, "*batch num_regimes state_dim state_dim"],
        *,
        validate_args: bool | None = None,
    ) -> None:
        if continuous_locs.ndim < 2:
            raise ValueError(
                "continuous_locs must have shape (*batch, num_regimes, state_dim)."
            )
        if continuous_covariances.ndim < 3:
            raise ValueError(
                "continuous_covariances must have shape "
                "(*batch, num_regimes, state_dim, state_dim)."
            )

        num_regimes = categorical_probs.shape[-1]
        state_dim = continuous_locs.shape[-1]
        if continuous_locs.shape[-2] != num_regimes:
            raise ValueError(
                "categorical_probs and continuous_locs disagree on "
                f"num_regimes: {num_regimes} != {continuous_locs.shape[-2]}."
            )
        if continuous_covariances.shape[-3:] != (
            num_regimes,
            state_dim,
            state_dim,
        ):
            raise ValueError(
                "continuous_covariances must end in "
                f"({num_regimes}, {state_dim}, {state_dim}); got "
                f"{continuous_covariances.shape[-3:]}."
            )

        batch_shape = lax.broadcast_shapes(
            categorical_probs.shape[:-1],
            continuous_locs.shape[:-2],
            continuous_covariances.shape[:-3],
        )
        probs = jnp.broadcast_to(categorical_probs, batch_shape + (num_regimes,))
        locs = jnp.broadcast_to(continuous_locs, batch_shape + (num_regimes, state_dim))
        covariances = jnp.broadcast_to(
            continuous_covariances,
            batch_shape + (num_regimes, state_dim, state_dim),
        )
        self.categorical_probs = probs
        self.continuous_locs = locs
        self.continuous_covariances = covariances
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(state_dim + 1,),
            validate_args=validate_args,
        )

    @property
    def num_regimes(self) -> int:
        """Number of discrete regimes."""
        return int(self.categorical_probs.shape[-1])

    @property
    def continuous_state_dim(self) -> int:
        """Dimension of the continuous part of the state."""
        return int(self.continuous_locs.shape[-1])

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Real[Array, "*sample *batch joint_state_dim"]:
        """Draw joint regime and continuous-state samples."""
        regime_key, state_key = jr.split(key)
        regimes = dist.Categorical(probs=self.categorical_probs).sample(
            regime_key, sample_shape
        )
        component_samples = dist.MultivariateNormal(
            self.continuous_locs,
            covariance_matrix=self.continuous_covariances,
        ).sample(state_key, sample_shape)
        state_indices = jnp.broadcast_to(
            regimes[..., None, None],
            component_samples.shape[:-2] + (1, self.continuous_state_dim),
        )
        states = jnp.take_along_axis(component_samples, state_indices, axis=-2)[
            ..., 0, :
        ]
        return jnp.concatenate(
            (regimes[..., None].astype(states.dtype), states),
            axis=-1,
        )

    def log_prob(
        self,
        value: Real[Array, "*sample *batch joint_state_dim"],
    ) -> Float[Array, "*sample *batch"]:
        """Evaluate the joint log density of ``[z, *x]``."""
        regimes = jnp.rint(value[..., 0]).astype(jnp.int32)
        continuous_state = value[..., 1:]
        regime_log_prob = dist.Categorical(probs=self.categorical_probs).log_prob(
            regimes
        )
        component_log_probs = dist.MultivariateNormal(
            self.continuous_locs,
            covariance_matrix=self.continuous_covariances,
        ).log_prob(continuous_state[..., None, :])
        continuous_log_prob = jnp.take_along_axis(
            component_log_probs, regimes[..., None], axis=-1
        )[..., 0]
        return regime_log_prob + continuous_log_prob


class RaoBlackwellizedParticleDistribution(dist.Distribution):
    r"""Gaussian-mixture posterior represented by Rao-Blackwellized particles.

    Each particle stores a discrete regime and a Gaussian conditional
    distribution for the continuous state. This distribution is used for
    filter-to-simulator posterior rollout; unlike a point-particle
    approximation, sampling retains the within-particle Gaussian uncertainty.

    Args:
        log_weights: Normalized or unnormalized particle log weights with shape
            ``(*batch, num_particles)``.
        regimes: Integer regime labels with shape
            ``(*batch, num_particles)``.
        continuous_locs: Particle-conditional means with shape
            ``(*batch, num_particles, state_dim)``.
        continuous_covariances: Particle-conditional covariances with shape
            ``(*batch, num_particles, state_dim, state_dim)``.
        validate_args: Whether NumPyro should validate samples and parameters.
    """

    arg_constraints: dict = {}
    support = constraints.real_vector
    pytree_data_fields = (
        "log_weights",
        "regimes",
        "continuous_locs",
        "continuous_covariances",
    )
    pytree_aux_fields = ("_batch_shape", "_event_shape")

    log_weights: Float[Array, "*batch num_particles"]
    regimes: Int[Array, "*batch num_particles"]
    continuous_locs: Float[Array, "*batch num_particles state_dim"]
    continuous_covariances: Float[Array, "*batch num_particles state_dim state_dim"]

    def __init__(
        self,
        log_weights: Float[Array, "*batch num_particles"],
        regimes: Int[Array, "*batch num_particles"],
        continuous_locs: Float[Array, "*batch num_particles state_dim"],
        continuous_covariances: Float[
            Array, "*batch num_particles state_dim state_dim"
        ],
        *,
        validate_args: bool | None = None,
    ) -> None:
        state_dim = continuous_locs.shape[-1]
        num_particles = continuous_locs.shape[-2]
        if regimes.shape[-1] != num_particles or log_weights.shape[-1] != num_particles:
            raise ValueError("All RBPF inputs must agree on num_particles.")
        if continuous_covariances.shape[-3:] != (
            num_particles,
            state_dim,
            state_dim,
        ):
            raise ValueError(
                "continuous_covariances must end in "
                f"({num_particles}, {state_dim}, {state_dim})."
            )
        batch_shape = lax.broadcast_shapes(
            log_weights.shape[:-1],
            regimes.shape[:-1],
            continuous_locs.shape[:-2],
            continuous_covariances.shape[:-3],
        )
        weights = jnp.broadcast_to(log_weights, batch_shape + (num_particles,))
        regimes = jnp.broadcast_to(regimes, batch_shape + (num_particles,))
        locs = jnp.broadcast_to(
            continuous_locs, batch_shape + (num_particles, state_dim)
        )
        covariances = jnp.broadcast_to(
            continuous_covariances,
            batch_shape + (num_particles, state_dim, state_dim),
        )
        self.log_weights = jax.nn.log_softmax(weights, axis=-1)
        self.regimes = regimes
        self.continuous_locs = locs
        self.continuous_covariances = covariances
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(state_dim + 1,),
            validate_args=validate_args,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Real[Array, "*sample *batch joint_state_dim"]:
        """Draw a particle, then draw its conditional continuous state."""
        particle_key, state_key = jr.split(key)
        particle_indices = dist.Categorical(logits=self.log_weights).sample(
            particle_key, sample_shape
        )
        component_samples = dist.MultivariateNormal(
            self.continuous_locs,
            covariance_matrix=self.continuous_covariances,
        ).sample(state_key, sample_shape)
        state_indices = jnp.broadcast_to(
            particle_indices[..., None, None],
            component_samples.shape[:-2] + (1, self.event_shape[0] - 1),
        )
        states = jnp.take_along_axis(component_samples, state_indices, axis=-2)[
            ..., 0, :
        ]
        regimes = jnp.take_along_axis(
            jnp.broadcast_to(
                self.regimes,
                sample_shape + self.regimes.shape,
            ),
            particle_indices[..., None],
            axis=-1,
        )[..., 0]
        return jnp.concatenate(
            (regimes[..., None].astype(states.dtype), states),
            axis=-1,
        )

    def log_prob(
        self,
        value: Real[Array, "*sample *batch joint_state_dim"],
    ) -> Float[Array, "*sample *batch"]:
        """Evaluate the particle-mixture density of ``[z, *x]``."""
        regime = jnp.rint(value[..., 0]).astype(self.regimes.dtype)
        continuous_state = value[..., 1:]
        component_log_probs = dist.MultivariateNormal(
            self.continuous_locs,
            covariance_matrix=self.continuous_covariances,
        ).log_prob(continuous_state[..., None, :])
        regime_matches = regime[..., None] == self.regimes
        joint_component_log_probs = jnp.where(
            regime_matches,
            self.log_weights + component_log_probs,
            -jnp.inf,
        )
        return jax.scipy.special.logsumexp(joint_component_log_probs, axis=-1)


__all__ = ["MixedStateDistribution"]
