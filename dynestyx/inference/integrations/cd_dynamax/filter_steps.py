"""Bind CD-Dynamax Gaussian moment routines to the shared filter scan."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from dynestyx.inference.filter_types import FilterInputs, Initialize, Predict, Update

type History = tuple[Array, dict[str, Array]]
type Moments = tuple[Array, Array]
type GaussianPrior = tuple[Moments, Array]
type GaussianState = tuple[Moments, Array, dict[str, Array]]


@dataclass(frozen=True)
class CDFilter[P, S, Posterior]:
    """Split steps and their projection into a native CD-Dynamax posterior."""

    initialize: Initialize[P]
    predict: Predict[P, S]
    update: Update[P, S]
    gather_outputs: Callable[[P, S], History]
    posterior_type: Callable[..., Posterior]
    forecast: Callable[[Moments, FilterInputs], Moments] | None = None
    ensemble_keys: bool = False

    def run(self, inputs: FilterInputs, key: PRNGKeyArray | None) -> Posterior:
        from dynestyx.inference.custom_filter import run_filter_steps, run_split_steps

        if key is not None and jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jax.random.key_data(key)
        if self.ensemble_keys:
            # CD-EnKF pairs analysis at t with the forecast from t to t+1.
            if key is None:
                raise ValueError("CD-EnKF requires a PRNG key.")
            keys = jax.random.split(key, len(inputs.y) + 1)
            pairs = jax.vmap(lambda k: jax.random.split(k, 2))(keys[1:])
            ll, fields = run_split_steps(
                self,
                inputs,
                self.gather_outputs,
                keys[0],
                jnp.roll(pairs[:, 0], 1, axis=0),
                pairs[:, 1],
            )
        else:
            ll, fields = run_filter_steps(self, inputs, key, self.gather_outputs)
        if self.forecast is not None:
            # Native discrete EKF/UKF histories end with a terminal forecast.
            last = jax.tree.map(lambda xs: xs[-1], inputs)
            terminal = self.forecast(
                (fields["filtered_means"][-1], fields["filtered_covariances"][-1]),
                last._replace(u_prev=last.u),
            )
            for name, value in zip(
                ("predicted_means", "predicted_covariances"), terminal, strict=True
            ):
                fields[name] = jnp.concatenate((fields[name][1:], value[None]))
        return self.posterior_type(marginal_loglik=ll[-1], **fields)


def gaussian_filter[Posterior](
    initial: Moments,
    predict_moments: Callable[[Moments, FilterInputs], Moments],
    condition: Callable[
        [Moments, FilterInputs], tuple[Array, Moments, dict[str, Array]]
    ],
    posterior_type: Callable[..., Posterior],
    *,
    save_forecasts: bool = False,
) -> CDFilter[GaussianPrior, GaussianState, Posterior]:
    """condition returns the likelihood increment, filtered moments and diagnostics."""

    def initialize(x: FilterInputs, *, key: PRNGKeyArray | None) -> GaussianPrior:
        return initial, jnp.zeros((), dtype=initial[0].dtype)

    def predict(
        filtered: GaussianState, x: FilterInputs, *, key: PRNGKeyArray | None
    ) -> GaussianPrior:
        moments, ll, _ = filtered
        return predict_moments(moments, x), ll

    def update(
        prior: GaussianPrior, x: FilterInputs, *, key: PRNGKeyArray | None
    ) -> GaussianState:
        moments, ll = prior
        increment, filtered, diagnostics = condition(moments, x)
        return filtered, ll + increment, diagnostics

    def gather(prior: GaussianPrior, filtered: GaussianState) -> History:
        (mean, cov), ll, diagnostics = filtered
        fields = dict(filtered_means=mean, filtered_covariances=cov, **diagnostics)
        if save_forecasts:
            fields["predicted_means"], fields["predicted_covariances"] = prior[0]
        return ll, fields

    return CDFilter(
        initialize,
        predict,
        update,
        gather,
        posterior_type,
        forecast=predict_moments if save_forecasts else None,
    )
