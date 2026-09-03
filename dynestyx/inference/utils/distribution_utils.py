import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray, Real, Shaped
from numpyro.distributions import constraints

from dynestyx.inference.integrations.utils import (
    WeightedParticles,
    covariance_from_cholesky,
)
from dynestyx.inference.utils.plate_utils import (
    _slice_time_axis,
    _time_len_from_array,
)

MissingPolicy = Literal["raise", "empty"]


class _ForwardSimulationImproperUniform(dist.ImproperUniform):
    """An improper distribution sampled by dynamical forward simulation.

    The forward sampler is used only for initialization and forward execution.
    Density evaluation is inherited from ``ImproperUniform`` and is always
    zero; it does not evaluate the log density of the ``DynamicalModel``.
    """

    pytree_data_fields = ("forward_sampler", "sample_transform")
    has_rsample = True

    def __init__(
        self,
        forward_sampler: Callable[[PRNGKeyArray], Shaped[Array, "..."]]
        | dist.Distribution,
        *,
        event_shape: tuple[int, ...],
        sample_transform: Callable[[Shaped[Array, "..."]], Shaped[Array, "..."]]
        | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self.forward_sampler = forward_sampler
        self.sample_transform = sample_transform
        super().__init__(
            constraints.real,
            batch_shape=(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _sample_one(self, key: PRNGKeyArray) -> Shaped[Array, "..."]:
        sample = (
            self.forward_sampler.sample(key)
            if isinstance(self.forward_sampler, dist.Distribution)
            else self.forward_sampler(key)
        )
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return jnp.reshape(jnp.asarray(sample), self.event_shape)

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Shaped[Array, "..."]:
        if not sample_shape:
            return self._sample_one(key)

        n_samples = math.prod(sample_shape)
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(self._sample_one)(keys)
        return jnp.reshape(samples, sample_shape + self.event_shape)

    def rsample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Shaped[Array, "..."]:
        return self.sample(key, sample_shape=sample_shape)


def _handle_missing_gaussian_sequence(
    *,
    missing: MissingPolicy,
    missing_message: str | None,
) -> list[dist.Distribution]:
    if missing == "empty":
        return []
    if missing == "raise":
        raise ValueError(
            missing_message or "Gaussian means/covariances were unavailable."
        )
    raise ValueError(f"Unknown missing Gaussian sequence policy: {missing!r}.")


def _gaussian_sequence_to_dists(
    means: Float[Array, "*plate time state_dim"] | None,
    covariances: Float[Array, "*plate time state_dim state_dim"] | None,
    *,
    plate_shapes: tuple[int, ...] = (),
    missing: MissingPolicy = "raise",
    missing_message: str | None = None,
) -> list[dist.Distribution]:
    """Convert time-indexed Gaussian parameters to per-time distributions."""
    if means is None or covariances is None:
        return _handle_missing_gaussian_sequence(
            missing=missing,
            missing_message=missing_message,
        )

    t_len = _time_len_from_array(means, plate_shapes)
    return [
        dist.MultivariateNormal(
            _slice_time_axis(means, t, plate_shapes),
            covariance_matrix=_slice_time_axis(covariances, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _check_if_ensemble_low_rank(
    ensemble: Real[Array, "... n_particles state_dim"],
) -> bool:
    r"""Whether the ensemble sample covariance is rank deficient.

    $P = X'X'^{\top}$ has rank at most $N-1$, so it is
    singular exactly when ``n_particles - 1 < state_dim``.
    Used to determine whether to use a low-rank representation of the covariance
    or to expand it to a dense matrix for `MultivariateNormal`.
    """
    n_particles, state_dim = ensemble.shape[-2], ensemble.shape[-1]
    return n_particles - 1 < state_dim


def _ensemble_sequence_to_low_rank_gaussian_dists(
    ensemble: Real[Array, "*plate time n_particles state_dim"],
    *,
    covariance_jitter: float = 0.0,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    r"""Convert an ensemble to per-time Gaussians with low-rank covariance representation.

    The ensemble sample covariance is a low-rank object:
    $$
    P_t = \frac{1}{N-1}\sum_{i}\left(x_t^{(i)}-\bar x_t\right)
                              \left(x_t^{(i)}-\bar x_t\right)^{\top}
        = X'_t X_t'^{\top},
    \qquad \operatorname{rank} P_t \le N-1,
    $$

    The low-rank representation is a $(\text{state\_dim}, N)$ factor $X'_t$
    (not expanded into a dense $(\text{state\_dim}, \text{state\_dim})$ matrix).

    ``covariance_jitter`` is the $\epsilon$ of $P_t + \epsilon I$.
    Default of ``0.0``, the distributions have the exact covariance, but no Lebesgue density.

    Note:
        `LowRankMultivariateNormal.log_prob` will yield ``nan`` unless a positive
        `covariance_jitter` is provided. Hence only use this for genuinely rank deficient matrices;
        otherwise see `_cholesky_state_sequence_to_dists`.

    Args:
        ensemble: Ensemble states, ``(*plate, time, n_particles, state_dim)``.
        covariance_jitter: Nonnegative $\epsilon$ added to the covariance as
            $\epsilon I$.
        plate_shapes: Leading plate dimensions, as elsewhere in this module.

    Returns:
        One `numpyro.distributions.LowRankMultivariateNormal` per time index.
    """
    n_particles = ensemble.shape[-2]
    state_dim = ensemble.shape[-1]
    mean = jnp.mean(ensemble, axis=-2)
    # (..., time, state_dim, n_particles): the factor X', not the product X' X'^T.
    cov_factor = jnp.swapaxes(ensemble - mean[..., None, :], -1, -2) / jnp.sqrt(
        jnp.asarray(n_particles - 1, dtype=ensemble.dtype)
    )
    cov_diag = jnp.full((state_dim,), covariance_jitter, dtype=ensemble.dtype)

    t_len = _time_len_from_array(mean, plate_shapes)
    return [
        dist.LowRankMultivariateNormal(
            _slice_time_axis(mean, t, plate_shapes),
            _slice_time_axis(cov_factor, t, plate_shapes),
            cov_diag,
        )
        for t in range(t_len)
    ]


def _particle_sequence_to_dists(
    particles: Real[Array, "*plate time n_particles state_dim"]
    | Real[Array, "*plate time n_particles"],
    log_weights: Float[Array, "*plate time n_particles"],
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert time-indexed particle arrays to per-time weighted particles."""
    if particles.ndim == len(plate_shapes) + 2:
        particles = particles[..., None]

    normalized_log_weights = jax.nn.log_softmax(log_weights, axis=-1)
    t_len = _time_len_from_array(normalized_log_weights, plate_shapes)
    return [
        WeightedParticles(
            particles=_slice_time_axis(particles, t, plate_shapes),
            log_weights=_slice_time_axis(normalized_log_weights, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _posterior_sequence_to_dists(
    posterior,
    *,
    means_attr: str,
    covariances_attr: str,
    particle_mode: bool,
    plate_shapes: tuple[int, ...] = (),
    missing: MissingPolicy = "raise",
    missing_message: str | None = None,
) -> list[dist.Distribution]:
    """Convert a backend posterior object to per-time distributions."""
    if particle_mode:
        return _particle_sequence_to_dists(
            posterior.particles,
            posterior.log_weights,
            plate_shapes=plate_shapes,
        )

    return _gaussian_sequence_to_dists(
        getattr(posterior, means_attr),
        getattr(posterior, covariances_attr),
        plate_shapes=plate_shapes,
        missing=missing,
        missing_message=missing_message,
    )


def _cholesky_state_sequence_to_dists(
    states,
    *,
    particle_mode: bool,
    plate_shapes: tuple[int, ...] = (),
    covariance_jitter: float = 0.0,
) -> list[dist.Distribution]:
    r"""Convert cuthbert state objects to per-time distributions.

    Three state families are handled, dispatched structurally:

    - particle states (`.particles`, `.log_weights`) become `WeightedParticles`;
    - ensemble states (`.ensemble`, i.e. `EnKFState` and `EnRTSState`) become low-rank Gaussians,
    when the ensemble is rank-deficient (``n_particles - 1 < state_dim``);
    - everything else becomes a dense `MultivariateNormal`.

    ``covariance_jitter`` is applied in both Gaussian branches; see
    `_ensemble_sequence_to_low_rank_gaussian_dists`.
    """
    if particle_mode:
        return _particle_sequence_to_dists(
            states.particles,
            states.log_weights,
            plate_shapes=plate_shapes,
        )

    if hasattr(states, "ensemble") and _check_if_ensemble_low_rank(states.ensemble):
        return _ensemble_sequence_to_low_rank_gaussian_dists(
            states.ensemble,
            covariance_jitter=covariance_jitter,
            plate_shapes=plate_shapes,
        )

    covariances = covariance_from_cholesky(states.chol_cov)
    if covariance_jitter:
        covariances = covariances + covariance_jitter * jnp.eye(
            covariances.shape[-1], dtype=covariances.dtype
        )
    return _gaussian_sequence_to_dists(
        states.mean,
        covariances,
        plate_shapes=plate_shapes,
    )


def _categorical_log_probs_to_dists(
    log_probs: Float[Array, "*plate time n_states"],
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert time-indexed categorical log-probs to per-time distributions."""
    t_len = _time_len_from_array(log_probs, plate_shapes)
    return [
        dist.Categorical(probs=jnp.exp(_slice_time_axis(log_probs, t, plate_shapes)))
        for t in range(t_len)
    ]
