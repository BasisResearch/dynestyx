# Dynestyx in a Nutshell

![dynestyx logo](logo/dynestyx.gif)

`dynestyx` is a library for Bayesian modeling and inference of dynamical systems. It extends [NumPyro](https://num.pyro.ai/en/stable/) to provides state-of-the-art inference methods for state space models—with a clear separation between *what* the model is and *how* you simulate or infer it.

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

## Quick Example

**[Lorenz 63 with partial noisy observations](quick_example.ipynb)** — A notebook that:

1. Defines the Lorenz 63 SDE with parameters $\rho$, state noise $\sigma_{\text{diff}}$, and observation noise $\sigma_{\text{obs}}$
2. Simulates trajectories with `SDESimulator` (observing only $x_1$ with Gaussian noise)
3. Runs NUTS + EnKF inference to recover all 3 parameters

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

- **[A gentle introduction](tutorials/gentle_intro/00_index.ipynb)** — Multi-part tutorial from NumPyro and Bayesian workflow → discrete-time dynestyx → filtering and MLL → pseudomarginal inference → SVI → continuous-time (SDEs) → HMMs
- **[Tutorials](tutorials.md)** — Quickstart, discrete-time inference, SDE inference, HMM inference, ODE inference, and more

## See also

Other JAX-based libraries for dynamical systems:

- **[dynamax](https://github.com/probml/dynamax)** — Discrete-time state space models with linear/non-linear Kalman filters and Bayesian parameter estimation
- **[cd-dynamax](https://github.com/hd-UQ/cd_dynamax)** — Continuous-discrete state space models with EnKF, EKF, UKF, PF and Bayesian parameter estimation
- **[PFJax](https://pfjax.readthedocs.io/en/latest/)** — Nonlinear and non-Gaussian discrete-time models with particle filters and particle MCMC
- **[Cuthbert](https://state-space-models.github.io/cuthbert/)** — Discrete-time state space models with linear/non-linear Kalman (and Particle Filters) filters, options for associative scans.
- **[diffrax](https://docs.kidger.site/diffrax/)** - Numerical differential equation solvers.

Other probabilistic programming languages with support for dynamical systems:

- **[STAN](https://mc-stan.org)** — Probabilistic programming with Hamiltonian Monte Carlo
    - [ODEs in Stan](https://mc-stan.org/docs/stan-users-guide/odes.html) — ODEs are a special transformation requiring little extra treatment from the user
- **[NumPyro](https://num.pyro.ai/en/stable/)** — JAX-based probabilistic programming
    - [ODEs in NumPyro](https://num.pyro.ai/en/stable/tutorials/lotka_volterra_multiple.html) — ODE solver must be defined inside the model (violates separation of concerns)
    - [SDEs in NumPyro](https://num.pyro.ai/en/stable/distributions.html#eulermaruyama) — `dist.EulerMaruyama` infers every Gaussian increment from an Euler–Maruyama solver
- **[ChiRho](https://basisresearch.github.io/chirho/)** — Probabilistic programming with causal tooling
    - [ODEs in ChiRho](https://basisresearch.github.io/chirho/dynamical_multi.html) — Hierarchical parameter inference in ODEs
