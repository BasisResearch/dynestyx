import dataclasses
import math

import jax
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
from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build in_axes for a pytree based on whether leaves have plate batch dims.

    A leaf is considered batched if it is a JAX array whose leading dimensions
    match the plate_shapes tuple. Batched leaves get in_axes=0; others get None.
    The same in_axes can be reused for each nested vmap call since each vmap
    peels off one leading dimension.
    """
    n_plates = len(plate_shapes)

    def _axis(leaf):
        if not isinstance(leaf, jax.Array) or leaf.ndim < n_plates:
            return None
        for i, ps in enumerate(plate_shapes):
            if leaf.shape[i] != ps:
                return None
        return 0

    return jax.tree.map(_axis, tree)


def _array_plate_axis(arr, plate_shapes: tuple[int, ...]):
    """Return 0 if arr has leading dims matching plate_shapes, else None."""
    if arr is None:
        return None
    n_plates = len(plate_shapes)
    if arr.ndim < n_plates:
        return None
    for i, ps in enumerate(plate_shapes):
        if arr.shape[i] != ps:
            return None
    return 0


def _tree_has_axis_zero(tree) -> bool:
    """Return True if any pytree leaf has in_axes=0."""
    return any(axis == 0 for axis in jax.tree.leaves(tree))


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
    for leaf in jax.tree.leaves(dynamics):
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
    has_batched_dynamics = _tree_has_axis_zero(dyn_axes)
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
    ) -> None:
        """Compute batched marginal log-likelihoods via vmap for plate contexts.

        Vmaps the pure-JAX compute function over each plate dimension, then issues
        a single numpyro.factor with the batched log-likelihoods. The enclosing
        numpyro.plate context handles summation.

        Returns None (no filtered distributions for batched case).
        """
        # Determine the compute function (dispatch before vmap).
        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )

            def compute_loglik(dyn, ot, ov, ct, cv, k):
                filtered = compute_continuous_filter(
                    dyn,
                    config,
                    k,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_times=ct,
                    ctrl_values=cv,
                )
                return filtered.marginal_loglik

        elif isinstance(config, HMMConfigs):

            def compute_loglik(dyn, ot, ov, ct, cv, k):
                loglik, _ = compute_hmm_filter(
                    dyn,
                    obs_times=ot,
                    obs_values=ov,
                    ctrl_values=cv,
                )
                return loglik

        elif isinstance(config, DiscreteTimeConfigs):
            if config.filter_source == "cuthbert":

                def compute_loglik(dyn, ot, ov, ct, cv, k):
                    loglik, _ = compute_cuthbert_filter(
                        dyn,
                        config,
                        k,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )
                    return loglik

            elif config.filter_source == "cd_dynamax":

                def compute_loglik(dyn, ot, ov, ct, cv, k):
                    posterior = compute_cd_dynamax_discrete_filter(
                        dyn,
                        config,
                        obs_times=ot,
                        obs_values=ov,
                        ctrl_times=ct,
                        ctrl_values=cv,
                    )
                    return posterior.marginal_loglik

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
        vmapped = compute_loglik
        for _ in plate_shapes:
            vmapped = jax.vmap(
                vmapped,
                in_axes=(dyn_axes, ot_axis, ov_axis, ct_axis, cv_axis, k_axis),
            )

        marginal_logliks = vmapped(
            dynamics,
            obs_times,
            obs_values,
            ctrl_times,
            ctrl_values,
            keys,
        )

        numpyro.factor(f"{name}_marginal_log_likelihood", marginal_logliks)
        numpyro.deterministic(f"{name}_marginal_loglik", marginal_logliks)
        return None


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
