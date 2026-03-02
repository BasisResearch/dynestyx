# Dynestyx in a Nutshell

![dynestyx logo](logo/dynestyx.gif)

`dynestyx` is a library for Bayesian modeling and inference of dynamical systems. It extends [NumPyro](https://num.pyro.ai/en/stable/) to provide state-of-the-art inference methods for state space models—with a clear separation between *what* the model is and *how* you simulate or infer it.

Why `dynestyx`? It seamlessly wraps our favorite ways to learn dynamics from messy time-series data (and there are many!) in a principled NumPyro Bayesian workflow. The engines under-the-hood address noise, partial observations, irregular samples, uncertainties, and just about any model class you want to try out! Support for multiple trajectories and hierarchical inference coming soon! Don't see your favorite methods? Tell us about it---or better, contribute by submitting a Pull Request!

## Features

- **Unified API** — Discrete-time and continuous-time dynamical systems (SDEs, ODEs, HMMs) in one interface
- **Rich model class** - Define custom initial conditions, state evolution processes, and observation models
- **Decoupled model and inference** — Write your model once; choose simulators, filters, or MCMC/Variational inference independently
- **Multiple inference methods** — joint state-and-parameter inference (via NUTS or stochastic variational inference), filters for marginalization (KF, EnKF, EKF, UKF, Particle Filter), pseudo-marginal MCMC (particle filter or EnKF within NUTS), gradient-matching.
- **NumPyro integration** — Builds on NumPyro’s probabilistic programming primitives, handlers, and inference stack
- **JAX-based** — Fully JIT-compilable and GPU-compatible

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/BasisResearch/dynestyx.git
```

Or with `pip`:

```bash
pip install git+https://github.com/BasisResearch/dynestyx.git
```

## Quick Example: Simulation

Define a dynamical model, wrap it with a simulator, and generate synthetic trajectories by passing observation times (and optionally controls) as kwargs:

```python
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import dynestyx as dsx
from dynestyx import DynamicalModel, DiscreteTimeSimulator
from numpyro.infer import Predictive

def model(phi=None, obs_times=None, obs_values=None):
    phi = numpyro.sample("phi", dist.Uniform(0.0, 1.0), obs=phi)
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_n, t_next: dist.Normal(phi * x, 0.5),
        observation_model=lambda x, u, t: dist.Normal(0.0, jnp.exp(x / 2.0)),
    )
    return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

obs_times = jnp.arange(0.0, 100.0, 1.0)
with DiscreteTimeSimulator():
    samples = Predictive(model, num_samples=1)(jr.PRNGKey(0), phi=0.9, obs_times=obs_times)
```

## Quick Example: Inference

Using the simulated `samples` and `obs_times` from above, condition on the data and infer parameters with a filter plus NUTS (no explicit state sampling):

```python
from dynestyx import Filter
from dynestyx.inference.filters import ContinuousTimeEnKFConfig
from numpyro.infer import MCMC, NUTS

obs_values = samples["observations"][0]

def inference_model():
    with Filter(filter_config=ContinuousTimeEnKFConfig(n_particles=25)):
        return model(obs_times=obs_times, obs_values=obs_values)

mcmc = MCMC(NUTS(inference_model), num_warmup=100, num_samples=100)
mcmc.run(jr.PRNGKey(1))
posterior = mcmc.get_samples()
```

See the [Lorenz 63 notebook](quick_example.ipynb) for a full SDE example with partial noisy observations.

## Citation

If you use dynestyx in your research, please cite:

```bibtex
@software{dynestyx,
  author = {{Basis Research Institute}},
  title = {dynestyx: Bayesian inference for dynamical systems},
  year = {2025},
  url = {https://github.com/BasisResearch/dynestyx},
}
```

## Next Steps
- **[A mathematical introduction](math_intro.md)** — Clearly defines the mathematical and statistical problems that `dynestyx` allows you to address. It maps concepts/algorithms to relevant pieces of code.
- **[Tutorials](tutorials/gentle_intro/00_index.ipynb)** — Multi-part tutorial from NumPyro and Bayesian workflow → discrete-time dynestyx → filtering and MLL → pseudomarginal inference → SVI → continuous-time (SDEs) → HMMs
- **[Examples](tutorials.md)** — Quickstart, discrete-time inference, SDE inference, HMM inference, ODE inference, and more

## See also

Other JAX-based libraries for dynamical systems:

- **[dynamax](https://github.com/probml/dynamax)** — Discrete-time state space models with linear/non-linear Kalman filters and Bayesian parameter estimation
- **[cd-dynamax](https://github.com/hd-UQ/cd_dynamax)** — Continuous-discrete state space models with EnKF, EKF, UKF, PF and Bayesian parameter estimation
- **[PFJax](https://pfjax.readthedocs.io/en/latest/)** — Nonlinear and non-Gaussian discrete-time models with particle filters and particle MCMC
- **[Cuthbert](https://state-space-models.github.io/cuthbert/)** — Discrete-time state space models with linear/non-linear Kalman (and Particle Filters) filters, options for associative scans.
- **[diffrax](https://docs.kidger.site/diffrax/)** - Numerical differential equation solvers.

Other probabilistic programming languages with support for dynamical systems:

- **[Stan](https://mc-stan.org)** — Probabilistic programming with Hamiltonian Monte Carlo
    - [ODEs in Stan](https://mc-stan.org/docs/stan-users-guide/odes.html) — ODEs are a special transformation requiring little extra treatment from the user
- **[NumPyro](https://num.pyro.ai/en/stable/)** — JAX-based probabilistic programming
    - [ODEs in NumPyro](https://num.pyro.ai/en/stable/tutorials/lotka_volterra_multiple.html) — ODE solver must be defined inside the model (violates separation of concerns)
    - [SDEs in NumPyro](https://num.pyro.ai/en/stable/distributions.html#eulermaruyama) — `dist.EulerMaruyama` infers every Gaussian increment from an Euler–Maruyama solver
- **[ChiRho](https://basisresearch.github.io/chirho/)** — Probabilistic programming with causal tooling
    - [ODEs in ChiRho](https://basisresearch.github.io/chirho/dynamical_multi.html) — Hierarchical parameter inference in ODEs
- **[PyMC](https://www.pymc.io/welcome.html)** - Probabilistic programming in Python
    - [SSMs in PyMC](https://www.pymc.io/projects/extras/en/latest/statespace/generated/pymc_extras.statespace.core.PyMCStateSpace.html) - doc page in `pymc_extras`.
    - [Hurricane forecasting example in PyMC](https://www.pymc.io/projects/examples/en/latest/case_studies/ssm_hurricane_tracking.html) - Requires manual discretization of continuous-time systems, and limited support for non-linear systems. They write "Hopefully, someday the `StateSpace` module in `pymc-extras` may support non-linear state space specifications with either the Extended Kalman Filter or with the Unscented Kalman Filter."
