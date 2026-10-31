"""Persistent observation-layout ownership for NumPyro model callables."""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Self

from jaxtyping import PRNGKeyArray
from numpyro.handlers import seed

from dynestyx.observation_missingness import MissingObservationMetadata

_MissingnessCache = dict[tuple[str, tuple[int, ...]], MissingObservationMetadata]
_active_missingness_cache: ContextVar[_MissingnessCache | None] = ContextVar(
    "dynestyx_prepared_model_missingness", default=None
)


def _get_prepared_missingness_cache() -> _MissingnessCache | None:
    """Return the innermost wrapped model's cache, if one is active."""
    return _active_missingness_cache.get()


class PreparedModel:
    """Wrap a NumPyro model with a persistent LPB missingness-layout cache.

    Automatic and explicit LatentPathBuilder handlers share this wrapper's
    metadata by sample-site name and observation shape, including plate-member
    names. Separate wrappers have independent caches. Dynamics may be rebuilt
    from sampled parameters on every execution.

    Call ``prepare`` with concrete observations before passing observations
    dynamically through JIT or ``MCMC(..., jit_model_args=True)``. An ordinary
    eager call under a NumPyro seed also prepares layouts. A first call with
    dynamically traced observations needs a prepared layout or explicit
    ``missing_obs_metadata``.

    Observation values and times may change, but the missingness pattern for
    an existing site and shape must remain fixed. Use a new wrapper and a new
    inference/compiled callable for a different pattern. Explicit metadata
    overrides the wrapper's layout for that call without changing its cache.

    Attributes:
        model: Original NumPyro model callable. Only static missingness metadata
            is retained; parameters, inferred states, and execution outputs are
            not cached.
    """

    def __init__(self, model: Callable[..., Any]) -> None:
        """Create an independent cache for ``model`` without executing it."""
        self.model = model
        self._missingness_cache: _MissingnessCache = {}
        functools.update_wrapper(self, model, updated=())

    def _call_with_cache(
        self, cache: _MissingnessCache, *args: Any, **kwargs: Any
    ) -> Any:
        token = _active_missingness_cache.set(cache)
        try:
            return self.model(*args, **kwargs)
        finally:
            _active_missingness_cache.reset(token)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the model with this wrapper's layouts available to LPB."""
        return self._call_with_cache(self._missingness_cache, *args, **kwargs)

    def prepare(
        self, *args: Any, rng_seed: int | PRNGKeyArray = 0, **kwargs: Any
    ) -> Self:
        """Run the model eagerly to prepare layouts, then return this wrapper.

        Args:
            *args: Positional model arguments, containing concrete observations.
            rng_seed: NumPyro seed or key for this preparation execution.
                Defaults to zero; subsequent inference supplies its own seed.
            **kwargs: Keyword model arguments.

        The full model executes once, including its usual side effects and
        handlers. Execution outputs are discarded. Failed preparation leaves
        the existing cache unchanged. This method does not reset an existing
        layout or invalidate JAX's compiled functions.
        """
        cache = dict(self._missingness_cache)
        with seed(rng_seed=rng_seed):
            self._call_with_cache(cache, *args, **kwargs)
        self._missingness_cache = cache
        return self
