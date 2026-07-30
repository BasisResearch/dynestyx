"""Register NumPyro sites for filter and smoother outputs."""

import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Float, Real

from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    HMMConfig,
    PFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.configs.smoother import (
    BaseSmootherConfig,
    PFSmootherConfig,
    _config_to_smoother_record_kwargs,
)
from dynestyx.inference.integrations.utils import covariance_from_cholesky
from dynestyx.utils import _should_record_field


def register_filter_sites(
    name: str,
    marginal_loglik: Real[Array, "*plate"],
    states: object,
    filter_config: BaseFilterConfig,
) -> None:
    """Register the likelihood and requested outputs from a filter run.

    The marginal log likelihood is recorded as a factor in the NumPyro model
    and deterministic site. Other fields are recorded as a deterministic site
    if they are marked as `True` in the corresponding `record_*` field of the config,
    or if their size does not exceed `record_max_elems` and `record_*` is `None`.

    Args:
        name: Prefix for each NumPyro site name.
        marginal_loglik: Marginal log likelihood computed by the filter.
        states: Backend result containing the filtered states.
        filter_config: Filter configuration that identifies the result format
            and selects the outputs to record.

    Returns:
        None.
    """
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    if isinstance(filter_config, HMMConfig):
        return

    record_kwargs = _config_to_record_kwargs(filter_config)

    if isinstance(filter_config, tuple(ContinuousTimeConfigs)):
        _add_continuous_filter_sites(name, states, record_kwargs)
    elif isinstance(filter_config, PFConfig):
        _add_cuthbert_pf_sites(name, states, record_kwargs)
    else:
        _add_gaussian_filter_sites(name, states, filter_config, record_kwargs)


def register_hmm_filter_sites(
    name: str,
    loglik: Real[Array, "*plate"],
    log_filt_seq: Float[Array, "*plate time n_states"],
    filter_config: HMMConfig,
) -> None:
    """Register the likelihood and requested outputs from an HMM filter run.

    The `loglik` is recorded as a factor and deterministic site. Other fields
    are recorded as a deterministic site if they are marked as `True` in the
    corresponding `record_*` field of the config, or if their size does not
    exceed `record_max_elems` and `record_*` is `None`.

    Args:
        name: Prefix for each NumPyro site name.
        loglik: Marginal log likelihood computed by the HMM filter.
        log_filt_seq: Log filtering probabilities at each time.
        filter_config: HMM configuration that selects the outputs to record.

    Returns:
        None.
    """
    numpyro.factor(f"{name}_marginal_log_likelihood", loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", loglik)

    record_max_elems = filter_config.record_max_elems

    if _should_record_field(
        filter_config.record_log_filtered, log_filt_seq.shape, record_max_elems
    ):
        numpyro.deterministic(f"{name}_log_filtered_states", log_filt_seq)

    if _should_record_field(
        filter_config.record_filtered, log_filt_seq.shape, record_max_elems
    ):
        numpyro.deterministic(f"{name}_filtered_states", jnp.exp(log_filt_seq))


def register_smoother_sites(
    name: str,
    marginal_loglik: Real[Array, "*plate"],
    states: object,
    smoother_config: BaseSmootherConfig,
) -> None:
    """Register the likelihood and requested outputs from a smoother run.

    The marginal log likelihood is recorded as a factor in the NumPyro model
    and deterministic site. Other fields are recorded as a deterministic site
    if they are marked as `True` in the corresponding `record_*` field of the config,
    or if their size does not exceed `record_max_elems` and `record_*` is `None`.

    Args:
        name: Prefix for each NumPyro site name.
        marginal_loglik: Marginal log likelihood computed during smoothing.
        states: Backend result containing the smoothed states.
        smoother_config: Smoother configuration that identifies the result
            format and selects the outputs to record.

    Returns:
        None.
    """
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    record_kwargs = _config_to_smoother_record_kwargs(smoother_config)

    if isinstance(smoother_config, PFSmootherConfig):
        _add_cuthbert_pf_smoother_sites(name, states, record_kwargs)
    elif hasattr(states, "smoothed_means"):
        _add_cd_dynamax_smoother_sites(name, states, record_kwargs)
    elif states is not None and hasattr(states, "mean"):
        _add_cuthbert_gaussian_smoother_sites(name, states, record_kwargs)


def _add_continuous_filter_sites(name: str, filtered, record_kwargs: dict) -> None:
    """Register requested outputs from a continuous-time Gaussian filter.

    Args:
        name: Prefix for each NumPyro site name.
        filtered: Backend result with `filtered_means` and
            `filtered_covariances` arrays.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    max_elems = record_kwargs["record_max_elems"]
    means_shape = filtered.filtered_means.shape
    cov_shape = filtered.filtered_covariances.shape
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], means_shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"], cov_shape, max_elems
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"],
        (cov_shape[0], cov_shape[1]),
        max_elems,
    )
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", filtered.filtered_means)
    if add_cov:
        numpyro.deterministic(
            f"{name}_filtered_states_cov", filtered.filtered_covariances
        )
    if add_cov_diag:
        diag_cov = jnp.diagonal(filtered.filtered_covariances, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)


def _add_cuthbert_pf_sites(name: str, states, record_kwargs: dict) -> None:
    """Register requested outputs from a Cuthbert particle filter.

    If a mean or covariance is requested, the function computes weighted
    moments from the particles and their log weights. Scalar-state particles
    receive a trailing state dimension before they are recorded.

    Args:
        name: Prefix for each NumPyro site name.
        states: Particle-filter result with `particles` and `log_weights`
            arrays.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    log_weights = states.log_weights
    particles = states.particles
    if particles.ndim == 2:
        particles = particles[..., None]
    max_elems = record_kwargs["record_max_elems"]
    t_len, n_particles, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_filtered_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_filtered_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], (t_len, state_dim), max_elems
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t_len, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t_len, state_dim), max_elems
    )

    need_filtered_means = (
        add_mean or add_filtered_states_cov or add_filtered_states_cov_diag
    )

    if need_filtered_means:
        w = jax.nn.softmax(log_weights, axis=1)[..., None]
        filtered_means = jnp.sum(particles * w, axis=1)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, w.squeeze(-1)
        )
        filtered_covariances = second_mom - jnp.einsum(
            "...tj,...tk->...tjk", filtered_means, filtered_means
        )

    if add_particles:
        numpyro.deterministic(f"{name}_filtered_particles", particles)
    if add_log_weights:
        numpyro.deterministic(f"{name}_filtered_log_weights", log_weights)
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", filtered_means)
    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_covariances)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_covariances, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)


def _add_gaussian_filter_sites(
    name: str, states, filter_config: BaseFilterConfig, record_kwargs: dict
) -> None:
    """Register requested outputs from a discrete-time Gaussian filter.

    Results from cd-dynamax expose `filtered_means` and
    `filtered_covariances`. Cuthbert results expose `mean` and `chol_cov`; this
    function derives the full covariance or its diagonal only when requested.

    Args:
        name: Prefix for each NumPyro site name.
        states: Gaussian-filter result in the format returned by its backend.
        filter_config: Configuration that selected the Gaussian registration
            path.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    max_elems = record_kwargs["record_max_elems"]

    if hasattr(states, "filtered_means"):
        means = states.filtered_means
        covs = states.filtered_covariances
        if means is None:
            return
        t_len, state_dim = means.shape
        add_mean = _should_record_field(
            record_kwargs["record_filtered_states_mean"], means.shape, max_elems
        )
        add_cov = _should_record_field(
            record_kwargs["record_filtered_states_cov"],
            (t_len, state_dim, state_dim),
            max_elems,
        )
        add_cov_diag = _should_record_field(
            record_kwargs["record_filtered_states_cov_diag"],
            (t_len, state_dim),
            max_elems,
        )
        if add_mean:
            numpyro.deterministic(f"{name}_filtered_states_mean", means)
        if add_cov and covs is not None:
            numpyro.deterministic(f"{name}_filtered_states_cov", covs)
        if add_cov_diag and covs is not None:
            diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
            numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
    elif hasattr(states, "mean"):
        mean = states.mean
        chol_cov = states.chol_cov
        t_len, state_dim, _ = chol_cov.shape

        add_mean = _should_record_field(
            record_kwargs["record_filtered_states_mean"], mean.shape, max_elems
        )
        add_chol_cov = _should_record_field(
            record_kwargs["record_filtered_states_chol_cov"],
            chol_cov.shape,
            max_elems,
        )
        add_filtered_states_cov = _should_record_field(
            record_kwargs["record_filtered_states_cov"],
            (t_len, state_dim, state_dim),
            max_elems,
        )
        add_filtered_states_cov_diag = _should_record_field(
            record_kwargs["record_filtered_states_cov_diag"],
            (t_len, state_dim),
            max_elems,
        )

        if add_mean:
            numpyro.deterministic(f"{name}_filtered_states_mean", mean)
        if add_chol_cov:
            numpyro.deterministic(f"{name}_filtered_states_chol_cov", chol_cov)

        if add_filtered_states_cov or add_filtered_states_cov_diag:
            filtered_cov = covariance_from_cholesky(chol_cov)

        if add_filtered_states_cov:
            numpyro.deterministic(f"{name}_filtered_states_cov", filtered_cov)
        if add_filtered_states_cov_diag:
            diag_cov = jnp.diagonal(filtered_cov, axis1=1, axis2=2)
            numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)


def _add_cuthbert_pf_smoother_sites(name: str, states, record_kwargs: dict) -> None:
    """Register requested outputs from a Cuthbert particle smoother.

    If a mean or covariance is requested, the function computes weighted
    moments from the particles and their log weights. Scalar-state particles
    receive a trailing state dimension before they are recorded.

    Args:
        name: Prefix for each NumPyro site name.
        states: Particle-smoother result with `particles` and `log_weights`
            arrays.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    log_weights = states.log_weights
    particles = states.particles
    if particles.ndim == 2:
        particles = particles[..., None]
    max_elems = record_kwargs["record_max_elems"]
    t1, _, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_smoothed_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_smoothed_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], (t1, state_dim), max_elems
    )
    add_smoothed_states_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_smoothed_states_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )

    need_means = add_mean or add_smoothed_states_cov or add_smoothed_states_cov_diag
    if need_means:
        w = jax.nn.softmax(log_weights, axis=1)[..., None]
        smoothed_means = jnp.sum(particles * w, axis=1)

    if add_smoothed_states_cov or add_smoothed_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, w.squeeze(-1)
        )
        smoothed_cov = second_mom - jnp.einsum(
            "...tj,...tk->...tjk", smoothed_means, smoothed_means
        )

    if add_particles:
        numpyro.deterministic(f"{name}_smoothed_particles", particles)
    if add_log_weights:
        numpyro.deterministic(f"{name}_smoothed_log_weights", log_weights)
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", smoothed_means)
    if add_smoothed_states_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", smoothed_cov)
    if add_smoothed_states_cov_diag:
        diag_cov = jnp.diagonal(smoothed_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


def _add_cd_dynamax_smoother_sites(name: str, posterior, record_kwargs: dict) -> None:
    """Register requested outputs from a cd-dynamax Gaussian smoother.

    No state sites are registered if the posterior does not contain both
    smoothed means and smoothed covariances.

    Args:
        name: Prefix for each NumPyro site name.
        posterior: Smoother result with `smoothed_means` and
            `smoothed_covariances` arrays.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    max_elems = record_kwargs["record_max_elems"]
    means = posterior.smoothed_means
    covs = posterior.smoothed_covariances
    if means is None or covs is None:
        return
    t1, state_dim = means.shape
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], means.shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", means)
    if add_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", covs)
    if add_cov_diag:
        diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


def _add_cuthbert_gaussian_smoother_sites(
    name: str, states, record_kwargs: dict
) -> None:
    """Register requested outputs from a Cuthbert Gaussian smoother.

    The function records the Cholesky covariance directly when requested. It
    derives the full covariance or its diagonal only when either is requested.

    Args:
        name: Prefix for each NumPyro site name.
        states: Gaussian-smoother result with `mean` and `chol_cov` arrays.
        record_kwargs: Resolved recording flags and maximum element count.

    Returns:
        None.
    """
    max_elems = record_kwargs["record_max_elems"]
    mean = states.mean
    chol_cov = states.chol_cov
    t1, state_dim, _ = chol_cov.shape

    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], mean.shape, max_elems
    )
    add_chol_cov = _should_record_field(
        record_kwargs["record_smoothed_states_chol_cov"],
        chol_cov.shape,
        max_elems,
    )
    add_smoothed_states_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_smoothed_states_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )

    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", mean)
    if add_chol_cov:
        numpyro.deterministic(f"{name}_smoothed_states_chol_cov", chol_cov)

    if add_smoothed_states_cov or add_smoothed_states_cov_diag:
        smoothed_cov = covariance_from_cholesky(chol_cov)

    if add_smoothed_states_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", smoothed_cov)
    if add_smoothed_states_cov_diag:
        diag_cov = jnp.diagonal(smoothed_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)
