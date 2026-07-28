# LatentPathBuilder

One way to do Bayesian inference on a dynamical system is to "unroll" it and perform joint inference. For example, consider a dynamical system 

$$
x_{t+1} \sim p(x_{t+1} \,|\, x_t)
$$ 

with observations

$$
y_t \sim p(y_t \,|\, x_t).
$$

Then we can perform inference on the joint distribution $p(x_{0:T}, \theta \,|\, y_{1:T})$ by considering a probabilistic program that computes recursions in a `for` loop, conceptually something like
```python
def model(ys):
  x_0 = numpyro.sample("x0", p_x0)
  theta = numpyro.sample("theta", p_theta)

  x_i = x_0

  for i in range(T):
    x_i = numpyro.sample(f"x_{i+1}", p_transition(x_i))
    numpyro.sample(f"y_{i+1}", p_observation(x_i), obs=ys[i])
```

`LatentPathBuilder` provides such a representation for joint learning, with support for arbitrary missingness in the data `y`. To support missingness and be more efficient, the pseudocode above *is not* how `LatentPathBuilder` is implemented, but is a useful mental model.

`LatentPathBuilder` is a NumPyro-facing handler: use it through `dsx.sample(...)` inside a
NumPyro model. For pure-JAX trajectory scoring of fixed latent values, use
`dsx.log_prob(...)` instead.

Conceptually:

- `Simulator` generates trajectories and observations.
- `LatentPathBuilder` constructs `state_path_params`, reconstructs the full
  `state_path`, and evaluates `log p(x, y | ...)`.
- `Filter` and `Smoother` remain the preferred handlers when observations should be marginalized.

The main output names are intentionally distinct from simulator rollout names:

- `f_state_path` / `f_state_path_times`: explicit latent-path inference outputs
  from `LatentPathBuilder`
- `f_states` / `f_times`: rollout outputs from `Simulator` and `dsx.simulate(...)`

## Implementation Details 

To support arbitrary missingness, and for efficiency, the actual implementation of the `LatentPathBuilder` differs from the simple "unrolling" mental model. In particular, the entire `state_path` is intiialized as a `numpyro` site, via an improper uniform prior. The improper uniform prior is modifying so that calling its `sample` method (for example, as used in `numpyro` MCMC samplers by default) provides draws from the actual SSM prior. In particular:

- discrete `state_path_params` are drawn from the SSM prior;
- ODE `state_path_params` are drawn from the initial-condition prior and then
  solved on the requested time grid;
- remaining `missing_obs_values` are drawn from the observation model given the simulated state path.

After simulation, the model is scored according to the joint log-likelihood $\log p(x, y | \theta)$, and registered as a `numpyro` site `f_joint_log_prob_factor`.

::: dynestyx.inference.latent.builder
    options:
      members:
        - LatentPathBuilder
