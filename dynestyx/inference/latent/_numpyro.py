"""Private NumPyro trace-registration helpers for latent-path inference.

The rest of the latent-path stack works in pure JAX. This module is the thin
bridge that turns those pure-JAX results into NumPyro sites when the user calls
``dsx.sample(...)`` in a NumPyro model.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array

from dynestyx.inference.latent.log_prob import TrajectoryLogProbTerms
from dynestyx.inference.latent.parameterization import AssembledStatePath


def base_latent_distribution(value: Array) -> dist.Distribution:
    """Return the dummy base distribution used for latent path construction.

    ``LatentPathBuilder`` represents latent variables through deterministic
    pure-JAX reconstruction and then corrects the NumPyro trace with a factor.
    The corresponding NumPyro sample sites therefore use a simple dummy base
    distribution with the right shape, whose log-probability is later
    subtracted back out.
    """
    param_shape = tuple(jnp.asarray(value).shape)
    base = dist.Normal(0.0, 1.0).expand(param_shape)
    return base.to_event(len(param_shape))


def build_latent_path_site_registrar(
    *,
    canonical_state_path_params: Array | None,
    canonical_missing_obs_values: Array | None,
    example_state_path_params: Array,
    example_missing_obs_values: Array | None,
    eager_assembled_state_path: AssembledStatePath | None,
    eager_log_prob_terms: TrajectoryLogProbTerms | None,
    evaluate_latent_values: Callable[
        [Array, Array | None], tuple[AssembledStatePath, TrajectoryLogProbTerms]
    ],
) -> Callable[[str], None]:
    """Build a deferred ``_register_numpyro_sites`` callback.

    The returned callback is attached to :class:`LatentStateResult`. When
    invoked, it:

    1. creates dummy NumPyro sample sites for latent blocks,
    2. either reuses an eagerly computed pure-JAX evaluation or recomputes one
       from sampled latent values,
    3. adds a correcting ``numpyro.factor`` equal to
       ``log p(x, y | ...) - log q_dummy(z)``, and
    4. registers deterministic sites exposing the reconstructed path and
       related metadata.

    Args:
        canonical_state_path_params:
            Concrete latent path parameters ``z`` when the caller provided them
            directly to ``dsx.sample(...)``. These are passed as ``obs=...`` to
            the dummy NumPyro sample site so the trace uses the supplied latent
            values rather than drawing fresh ones.
        canonical_missing_obs_values:
            Same idea as ``canonical_state_path_params``, but for the optional
            missing-observation augmentation latent block.
        example_state_path_params:
            A shape-only representative latent value used solely to define the
            dummy NumPyro distribution for ``state_path_params``. This is
            needed because on the ``dsx.sample(...)`` path we often do not yet
            know the actual latent value when we construct the registrar, but
            we still must know the event shape of the NumPyro sample site.
            In other words: ``canonical_*`` means "the actual value if we have
            it", whereas ``example_*`` means "a fake value with the right
            shape so NumPyro knows what kind of site to create".
        example_missing_obs_values:
            Shape-only representative value for the optional
            ``missing_obs_values`` latent block, used for the same reason as
            ``example_state_path_params``.
        eager_assembled_state_path:
            Reconstructed state path from an earlier pure-JAX evaluation, if
            one was already available before NumPyro registration.
        eager_log_prob_terms:
            Joint log-probability decomposition matching
            ``eager_assembled_state_path``. When both eager objects are present,
            the registrar can reuse them rather than recomputing ``x = g(z)``
            and ``log p(x, y | ...)`` after NumPyro samples the latent sites.
        evaluate_latent_values:
            Fallback pure-JAX callback used when eager results cannot be
            reused. It takes concrete latent-site values sampled/observed by
            NumPyro and returns the reconstructed path together with its
            trajectory log-probability terms.
    """
    state_path_param_base_dist = base_latent_distribution(example_state_path_params)
    missing_obs_base_dist = (
        None
        if example_missing_obs_values is None
        else base_latent_distribution(example_missing_obs_values)
    )
    can_reuse_eager_evaluation = (
        eager_assembled_state_path is not None
        and eager_log_prob_terms is not None
        and canonical_state_path_params is not None
        and (missing_obs_base_dist is None or canonical_missing_obs_values is not None)
    )

    def _register(site_name: str) -> None:
        state_path_param_site = numpyro.sample(
            f"{site_name}_state_path_params",
            state_path_param_base_dist,
            obs=canonical_state_path_params,
        )
        missing_obs_site = None
        if missing_obs_base_dist is not None:
            missing_obs_site = numpyro.sample(
                f"{site_name}_missing_obs_values",
                missing_obs_base_dist,
                obs=canonical_missing_obs_values,
            )

        if can_reuse_eager_evaluation:
            assembled_state_path = eager_assembled_state_path
            log_prob_terms = eager_log_prob_terms
            assert assembled_state_path is not None
            assert log_prob_terms is not None
        else:
            assembled_state_path, log_prob_terms = evaluate_latent_values(
                state_path_param_site, missing_obs_site
            )

        numpyro.factor(
            f"{site_name}_state_path_params_lp",
            log_prob_terms.joint_log_prob
            - state_path_param_base_dist.log_prob(state_path_param_site)
            - (
                0.0
                if missing_obs_site is None or missing_obs_base_dist is None
                else missing_obs_base_dist.log_prob(missing_obs_site)
            ),
        )
        numpyro.deterministic(
            f"{site_name}_state_path_param_times",
            assembled_state_path.state_path_param_times,
        )
        if assembled_state_path.state_path_param_coordinate_indices is not None:
            numpyro.deterministic(
                f"{site_name}_state_path_param_coordinate_indices",
                assembled_state_path.state_path_param_coordinate_indices,
            )
        numpyro.deterministic(
            f"{site_name}_state_path", assembled_state_path.state_path
        )
        numpyro.deterministic(
            f"{site_name}_state_path_times",
            assembled_state_path.state_path_times,
        )
        if log_prob_terms.missing_obs_times is not None:
            numpyro.deterministic(
                f"{site_name}_missing_obs_times",
                log_prob_terms.missing_obs_times,
            )
        if log_prob_terms.missing_obs_coordinate_indices is not None:
            numpyro.deterministic(
                f"{site_name}_missing_obs_coordinate_indices",
                log_prob_terms.missing_obs_coordinate_indices,
            )
        if log_prob_terms.completed_obs_values is not None:
            numpyro.deterministic(
                f"{site_name}_completed_obs_values",
                log_prob_terms.completed_obs_values,
            )
        numpyro.deterministic(
            f"{site_name}_joint_log_prob",
            log_prob_terms.joint_log_prob,
        )

    return _register


__all__ = ["build_latent_path_site_registrar"]
