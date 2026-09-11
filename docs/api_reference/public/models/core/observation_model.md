# ObservationModel

::: dynestyx.models.core.ObservationModel
    options:
      show_root_heading: false
      show_root_toc_entry: false

## masked_observation_log_prob

Score the observed marginal of a distribution returned by an observation model:

```python
import dynestyx as dsx

observation_dist = dynamics.observation_model(state, control, time)
log_likelihood = dsx.masked_observation_log_prob(
    observation_dist, y=observation, obs_mask=observed
)
```

This also works with plain callable observation models.

::: dynestyx.observation_missingness.masked_observation_log_prob
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Example

??? example "Negative Binomial observation model"
    ```python
    import jax
    import jax.numpy as jnp
    from numpyro import distributions as dist
    from dynestyx import ObservationModel


    class NegativeBinomialObservation(ObservationModel):
        def __init__(self, W: jnp.ndarray, alpha: float = 10.0):
            self.W = W
            self.alpha = alpha  # concentration/over-dispersion parameter

        def __call__(self, x, u, t):
            # log link: mean rate must stay positive
            mean = jnp.exp(self.W @ x)
            return dist.NegativeBinomial2(mean=mean, concentration=self.alpha)


    obs_model = NegativeBinomialObservation(
        W=jnp.array([[1.0, -0.5, 0.25]]),
        alpha=8.0,
    )

    dynamics = DynamicalModel(observation_model=obs_model, ...)

    ```
