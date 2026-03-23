import dataclasses
import math

import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    DiscreteTimeConfigs,
    EKFConfig,
    EnKFConfig,
    HMMConfig,
    HMMConfigs,
    KFConfig,
    PFConfig,
    PFResamplingConfig,
    UKFConfig,
)
from dynestyx.inference.hmm_filters import _filter_hmm, compute_hmm_filter
from dynestyx.inference.integrations.cd_dynamax.continuous import (
    compute_continuous_filter,
    run_continuous_filter,
)
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    compute_cd_dynamax_discrete_filter,
)
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    run_discrete_filter as run_cd_dynamax_discrete,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete,
)
from dynestyx.inference.integrations.utils import (
    WeightedParticles,
    particles_to_delta_mixtures,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime
from dynestyx.utils import _array_has_plate_dims, _ensure_continuous_bm_dim

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build in_axes for a pytree based on whether leaves have plate batch dims.

    A leaf is considered batched if it is a JAX array whose leading dimensions
    match the plate_shapes tuple. Batched leaves get in_axes=0; others get None.
    The same in_axes can be reused for each nested vmap call since each vmap
    peels off one leading dimension.
    """

    def _is_distribution_leaf(node) -> bool:
        return isinstance(node, numpyro.distributions.Distribution)

    def _axis(leaf):
        if not isinstance(leaf, jax.Array):
            return None
        if not _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=0):
            return None
        suffix_ndim = leaf.ndim - len(plate_shapes)
        # Map two classes of leaves:
        # 1) suffix_ndim >= 2: canonical batched tensors.
        # 2) suffix_ndim == 0: per-member scalar parameters captured in callable
        #    modules for nonlinear dynamics/observations.
        #
        # Keep suffix_ndim == 1 unmapped to avoid ambiguous false positives where
        # unbatched vectors/matrices happen to begin with a plate-sized dimension.
        return 0 if (suffix_ndim == 0 or suffix_ndim >= 2) else None

    return jax.tree.map(_axis, tree, is_leaf=_is_distribution_leaf)


def _array_plate_axis(arr, plate_shapes: tuple[int, ...]):
    """Return 0 if arr has leading dims matching plate_shapes, else None."""
    return 0 if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1) else None


def _leading_dims(arr: jax.Array | None, n_dims: int) -> tuple[int, ...] | None:
    """Return up to n_dims leading dimensions for diagnostics."""
    if arr is None:
        return None
    n = min(n_dims, arr.ndim)
    return tuple(int(d) for d in arr.shape[:n])


def _summarize_dynamics_leading_dims(
    dynamics: DynamicalModel, n_dims: int, max_items: int = 6
) -> str:
    """Summarize leading dimensions from JAX-array leaves in a model pytree."""
    shapes: list[tuple[int, ...]] = []
    for leaf in jax.tree.leaves(
        dynamics,
        is_leaf=lambda node: isinstance(node, numpyro.distributions.Distribution),
    ):
        if isinstance(leaf, jax.Array):
            n = min(n_dims, leaf.ndim)
            shapes.append(tuple(int(d) for d in leaf.shape[:n]))

    unique_shapes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            unique_shapes.append(shape)

    if len(unique_shapes) <= max_items:
        return str(unique_shapes)
    return f"{unique_shapes[:max_items]} (+{len(unique_shapes) - max_items} more)"


def _validate_batched_plate_alignment(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    dyn_axes,
    ot_axis: int | None,
    ov_axis: int | None,
    ct_axis: int | None,
    cv_axis: int | None,
    *,
    obs_times: jax.Array | None,
    obs_values: jax.Array | None,
    ctrl_times: jax.Array | None,
    ctrl_values: jax.Array | None,
) -> None:
    """Raise early when plate_shapes do not align with any batched input source."""
    has_batched_dynamics = any(axis == 0 for axis in jax.tree.leaves(dyn_axes))
    has_batched_data = any(axis == 0 for axis in (ot_axis, ov_axis, ct_axis, cv_axis))
    if has_batched_dynamics or has_batched_data:
        return

    n_plates = len(plate_shapes)
    diagnostics = (
        "Plate/data shape alignment failed before batched filtering. "
        f"plate_shapes={plate_shapes}. No dynamics leaves or observed/control arrays "
        "have leading dimensions matching plate_shapes. "
        f"dynamics_leading_dims={_summarize_dynamics_leading_dims(dynamics, n_plates)}; "
        f"obs_times_leading_dims={_leading_dims(obs_times, n_plates)}; "
        f"obs_values_leading_dims={_leading_dims(obs_values, n_plates)}; "
        f"ctrl_times_leading_dims={_leading_dims(ctrl_times, n_plates)}; "
        f"ctrl_values_leading_dims={_leading_dims(ctrl_values, n_plates)}. "
        "Expected at least one batched source to start with plate_shapes."
    )
    raise ValueError(diagnostics)


def _get_time_axis(plate_shapes: tuple[int, ...]) -> int:
    """Return the axis index corresponding to time after plate dimensions."""
    return len(plate_shapes)


def _time_len_from_array(arr: jax.Array, plate_shapes: tuple[int, ...]) -> int:
    """Infer sequence length from an array with plate dims followed by time."""
    return int(arr.shape[_get_time_axis(plate_shapes)])


def _slice_time_axis(
    arr: jax.Array, t: int, plate_shapes: tuple[int, ...]
) -> jax.Array:
    """Slice an array at time index t where time axis follows plate dims."""
    time_axis = _get_time_axis(plate_shapes)
    return arr[(slice(None),) * time_axis + (t, ...)]


def _cuthbert_states_to_dists(
    states,
    config: BaseFilterConfig,
    *,
    plate_shapes: tuple[int, ...],
) -> list[numpyro.distributions.Distribution]:
    """Convert vmapped cuthbert outputs to per-time filtered distributions."""
    if isinstance(config, PFConfig):
        particles = states.particles
        log_weights = states.log_weights
        # cuthbert includes an init step at index 0; align with dynestyx T convention.
        particles = particles[
            (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
        ]
        log_weights = log_weights[
            (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
        ]
        return _particle_to_batched_dists(
            particles,
            log_weights,
            plate_shapes=plate_shapes,
        )

    # Kalman / Taylor-KF variants expose mean/chol_cov and include init at index 0.
    mean = states.mean[(slice(None),) * len(plate_shapes) + (slice(1, None), ...)]
    chol_cov = states.chol_cov[
        (slice(None),) * len(plate_shapes) + (slice(1, None), ...)
    ]
    cov = jnp.matmul(chol_cov, jnp.swapaxes(chol_cov, -1, -2))
    t_len = _time_len_from_array(mean, plate_shapes)
    return [
        numpyro.distributions.MultivariateNormal(
            _slice_time_axis(mean, t, plate_shapes),
            covariance_matrix=_slice_time_axis(cov, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _posterior_to_dists(
    posterior,
    *,
    plate_shapes: tuple[int, ...],
    particle_mode: bool,
) -> list[numpyro.distributions.Distribution]:
    """Convert vmapped cd-dynamax posterior objects to per-time distributions."""
    if particle_mode:
        particles = posterior.particles
        log_weights = posterior.log_weights
        return _particle_to_batched_dists(
            particles,
            log_weights,
            plate_shapes=plate_shapes,
        )

    means = posterior.filtered_means
    covs = posterior.filtered_covariances
    if means is None or covs is None:
        raise ValueError(
            "Filtered means/covariances were unavailable for a Gaussian rollout path."
        )
    t_len = _time_len_from_array(means, plate_shapes)
    return [
        numpyro.distributions.MultivariateNormal(
            _slice_time_axis(means, t, plate_shapes),
            covariance_matrix=_slice_time_axis(covs, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _hmm_to_dists(
    log_filt_seq: jax.Array,
    *,
    plate_shapes: tuple[int, ...],
) -> list[numpyro.distributions.Distribution]:
    """Convert vmapped HMM filtered log-probs to Categorical distributions."""
    t_len = _time_len_from_array(log_filt_seq, plate_shapes)
    return [
        numpyro.distributions.Categorical(
            probs=jnp.exp(_slice_time_axis(log_filt_seq, t, plate_shapes))
        )
        for t in range(t_len)
    ]


def _particle_to_batched_dists(
    particles: jax.Array,
    log_weights: jax.Array,
    *,
    plate_shapes: tuple[int, ...],
) -> list[numpyro.distributions.Distribution]:
    """Build per-time plate-batched WeightedParticles from canonical PF outputs."""
    if particles.ndim == len(plate_shapes) + 2:
        particles = particles[..., None]

    # Flatten plate members -> use the canonical per-member helper from
    # dynestyx.inference.integrations.utils (main branch path).
    n_members = math.prod(plate_shapes) if plate_shapes else 1
    t_len = _time_len_from_array(log_weights, plate_shapes)
    part_tail = particles.shape[len(plate_shapes) :]
    w_tail = log_weights.shape[len(plate_shapes) :]
    flat_particles = particles.reshape((n_members, *part_tail))
    flat_log_weights = log_weights.reshape((n_members, *w_tail))
    per_member = [
        particles_to_delta_mixtures(flat_particles[i], flat_log_weights[i])
        for i in range(n_members)
    ]

    if not plate_shapes:
        return per_member[0]

    result: list[numpyro.distributions.Distribution] = []
    for t in range(t_len):
        logits_t = jnp.stack(
            [per_member[i][t].log_weights for i in range(n_members)],  # type: ignore[attr-defined]
            axis=0,
        ).reshape(*plate_shapes, -1)
        values_t = jnp.stack(
            [per_member[i][t].particles for i in range(n_members)],  # type: ignore[attr-defined]
            axis=0,
        ).reshape(*plate_shapes, *per_member[0][t].particles.shape)  # type: ignore[attr-defined]
        result.append(
            WeightedParticles(
                particles=values_t,
                log_weights=logits_t,
            )
        )
    return result


class BaseLogFactorAdder(ObjectInterpretation, HandlesSelf):
    """Base for filter handlers."""

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        filtered_dists = None
        if not (obs_times is None or obs_values is None):
            filtered_dists = self._add_log_factors(
                name,
                dynamics,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )

        # Filter consumes obs_times and obs_values, so they are passed forward as None
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=None,
            obs_values=None,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            filtered_times=obs_times,
            filtered_dists=filtered_dists,
            **kwargs,
        )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        # Inheritors should implement this method.
        raise NotImplementedError()


def _default_filter_config(dynamics: DynamicalModel):
    """Return appropriate default filter config when none specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()

    # default to particle filter in discrete time
    return EKFConfig(filter_source="cuthbert")


@dataclasses.dataclass
class Filter(BaseLogFactorAdder):
    r"""Performs Bayesian filtering to compute the filtering distribution $p(x_t | y_{1:t})$ and the marginal likelihood $\log p(y_{1:T})$.

    A `Filter` object should be used as a context manager around a call to a model with a `dsx.sample(...)` statement
    to condition a dynamical model on observations via a filtering algorithm. The filter
    is selected and dispatched according to the `filter_config` argument, which adds the
    marginal log-likelihood as a NumPyro factor, allowing for downstream parameter inference.

    Examples:
        >>> def model(obs_times=None, obs_values=None):
        ...     dynamics = DynamicalModel(...)
        ...     return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)
        >>> def filtered_model(t, y):
        ...     with Filter(filter_config=KFConfig()):
        ...         return model(obs_times=t, obs_values=y)

    What this does
    --------------
    Filtering is the recursive (potentially approximate) computation of the filtering distribution
    \(p(x_t \mid y_{1:t})\). It allows for the computation of the marginal likelihood:

    \[
      \log p(y_{1:T}) = \sum_{t=1}^T \log p(y_t \mid y_{1:t-1}),
    \]

    which in turn can be used to compute the posterior distribution over the parameters $p(\theta | y_{1:T})$.


    Available Filter Configurations
    ----------------------------------
    There are several different filters available in `dynestyx`, each with their own strengths and weaknesses.
    What filters are applicable to a given model depends heavily on any special structure of the model (for example, linear and/or Gaussian observations).
    For a summary table of all config classes and when to use them, see
    [Available filter configurations](../filter_configs.md).

    Defaults
    --------
    If `filter_config=None`, defaults are:

    - `ContinuousTimeEnKFConfig()` for continuous-time models, and
    - `EKFConfig(filter_source="cuthbert")` for discrete-time models.

    Notes:
        - If your latent state is *discrete* (an HMM), you must use `HMMConfig`.
        - What gets recorded to the trace (means/covariances, particles/weights,
        etc.) depends on `filter_config.record_*` and the backend implementation.

    Attributes:
        filter_config: Selects the filtering algorithm and its hyperparameters.
            If `None`, a reasonable default is chosen based on whether the model
            is continuous-time or discrete-time.
    """

    filter_config: BaseFilterConfig | None = None

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: jax.Array | None = None,
        obs_values: jax.Array | None = None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> list[numpyro.distributions.Distribution] | None:
        """
        Add the marginal log likelihood as a numpyro factor.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            plate_shapes: Tuple of plate sizes from enclosing dsx.plate contexts.
            obs_times: Observation times.
            obs_values: Observed values.
            ctrl_times: Control times (optional).
            ctrl_values: Control values (optional).
        """
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for filtering.")

        dynamics = _ensure_continuous_bm_dim(dynamics)

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )

        key = numpyro.prng_key() if config.crn_seed is None else config.crn_seed

        if plate_shapes:
            return self._add_log_factors_batched(
                name,
                dynamics,
                config,
                key=key,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            return _filter_continuous_time(
                name,
                dynamics,
                config,  # type: ignore[arg-type]
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )
        else:
            if isinstance(config, HMMConfigs):
                return _filter_hmm(
                    name,
                    dynamics,
                    config,  # type: ignore[arg-type]
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            elif isinstance(config, DiscreteTimeConfigs):
                return _filter_discrete_time(
                    name,
                    dynamics,
                    config,  # type: ignore[arg-type]
                    key=key,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            else:
                valid = [c.__name__ for c in HMMConfigs + DiscreteTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )

    def _add_log_factors_batched(
        self,
        name: str,
        dynamics: DynamicalModel,
        config: BaseFilterConfig,
        *,
        key: jax.Array | None,
        plate_shapes: tuple[int, ...],
        obs_times: jax.Array,
        obs_values: jax.Array,
        ctrl_times=None,
        ctrl_values=None,
    ) -> list[numpyro.distributions.Distribution]:
        """Compute batched marginal log-likelihoods via vmap for plate contexts.

        Vmaps the pure-JAX compute function over each plate dimension, issues one
        numpyro.factor with batched log-likelihoods, and reconstructs per-time
        filtered distributions with plate-shaped batch dimensions for rollout.
        """
        # Determine the compute function (dispatch before vmap).
        output_kind: str
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            output_kind = "continuous"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_continuous_filter(
                    dyn,
                    config,
                    k,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_times=ct,
                    ctrl_values=cv,
                )

        elif isinstance(config, HMMConfigs):
            output_kind = "hmm"

            def compute_output(dyn, ot, ov, ct, cv, k):
                return compute_hmm_filter(
                    dyn,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_values=cv,
                )

        elif isinstance(config, DiscreteTimeConfigs):
            if config.filter_source == "cuthbert":
                output_kind = "cuthbert"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cuthbert_filter(
                        dyn,
                        config,
                        k,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            elif config.filter_source == "cd_dynamax":
                output_kind = "cd_dynamax_discrete"

                def compute_output(dyn, ot, ov, ct, cv, k):
                    return compute_cd_dynamax_discrete_filter(
                        dyn,
                        config,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )

            else:
                raise ValueError(f"Unknown filter source: {config.filter_source}")
        else:
            raise ValueError(
                f"Unsupported filter config for plate: {type(config).__name__}"
            )

        # Pre-split keys for all plate members (needed for stochastic filters).
        if key is not None:
            total = math.prod(plate_shapes)
            keys = jax.random.split(key, total).reshape(*plate_shapes, -1)
        else:
            keys = None

        # Build in_axes: same axes reused for each nested vmap.
        dyn_axes = _make_plate_in_axes(dynamics, plate_shapes)
        ot_axis = _array_plate_axis(obs_times, plate_shapes)
        ov_axis = _array_plate_axis(obs_values, plate_shapes)
        ct_axis = _array_plate_axis(ctrl_times, plate_shapes)
        cv_axis = _array_plate_axis(ctrl_values, plate_shapes)
        _validate_batched_plate_alignment(
            dynamics,
            plate_shapes,
            dyn_axes,
            ot_axis,
            ov_axis,
            ct_axis,
            cv_axis,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        k_axis = 0 if keys is not None else None

        # Nest vmap for each plate dimension.
        vmapped = compute_output
        for _ in plate_shapes:
            vmapped = jax.vmap(
                vmapped,
                in_axes=(dyn_axes, ot_axis, ov_axis, ct_axis, cv_axis, k_axis),
            )

        outputs = vmapped(
            dynamics,
            obs_times,
            obs_values,
            ctrl_times,
            ctrl_values,
            keys,
        )

        if output_kind in {"continuous", "cd_dynamax_discrete"}:
            marginal_logliks = outputs.marginal_loglik
        elif output_kind == "hmm":
            marginal_logliks, log_filt_seq = outputs
        elif output_kind == "cuthbert":
            marginal_logliks, states = outputs
        else:
            raise ValueError(f"Unsupported batched output kind: {output_kind}")

        numpyro.factor(f"{name}_marginal_log_likelihood", marginal_logliks)
        numpyro.deterministic(f"{name}_marginal_loglik", marginal_logliks)

        if output_kind == "continuous":
            particle_mode = isinstance(config, ContinuousTimeDPFConfig)
            return _posterior_to_dists(
                outputs,
                plate_shapes=plate_shapes,
                particle_mode=particle_mode,
            )
        if output_kind == "cd_dynamax_discrete":
            return _posterior_to_dists(
                outputs,
                plate_shapes=plate_shapes,
                particle_mode=False,
            )
        if output_kind == "hmm":
            return _hmm_to_dists(
                log_filt_seq,
                plate_shapes=plate_shapes,
            )
        if output_kind == "cuthbert":
            return _cuthbert_states_to_dists(
                states,
                config,
                plate_shapes=plate_shapes,
            )

        raise ValueError(f"Unsupported batched output kind: {output_kind}")


def _filter_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Discrete-time marginal likelihood via cuthbert or cd-dynamax.

    Filter type inferred from config class: KFConfig, EKFConfig, UKFConfig (cd-dynamax)
    or EKFConfig (cuthbert), PFConfig (cuthbert).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """

    if filter_config.filter_source == "cd_dynamax":
        return run_cd_dynamax_discrete(
            name,
            dynamics,
            filter_config,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    elif filter_config.filter_source == "cuthbert":
        return run_cuthbert_discrete(
            name,
            dynamics,
            filter_config,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown filter source: {filter_config.filter_source}")


def _filter_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Continuous-time marginal likelihood via CD-Dynamax.

    Supports: EnKF, DPF, EKF, UKF (inferred from config type).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """
    return run_continuous_filter(
        name,
        dynamics,
        filter_config,  # type: ignore[arg-type]
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
    )


__all__ = [
    "ContinuousTimeKFConfig",
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeUKFConfig",
    "EKFConfig",
    "EnKFConfig",
    "Filter",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
