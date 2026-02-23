# ObservationModel

::: dynestyx.models.core.ObservationModel
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