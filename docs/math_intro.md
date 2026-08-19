# Introduction to `dynestyx`

In this introduction, we discuss some goals of `dynestyx`, and an overview of the types of problems it aims to solve. 

If you want to jump directly into code, take a look at the [Quick Example](tutorials/quickstart.ipynb), which shows how to build models in `dynestyx` and perform inference using observed data!

## Dynamical Systems and `dynestyx`

For the purposes of `dynestyx`, a dynamical system is a generative model depending on time. These generally come in the form of *state space models* (SSMs), where some unobserved state \(x_t \in \mathbb{R}^{d_x}\) evolves over time[^1], with occasional observations \(y_t \in \mathbb{R}^{d_y}\). Dynestyx allows the process that evolves \(x_t\), as well as the process that generates observations \(y_t\), to both be extremely general, and, in particular, allows for \(x_t\) to be specified as a continuous-time or discrete-time process. That is, \(x_t\) may be specified by a *stochastic differential equation* (SDE)[^2],

$$
    dx_t = \underbrace{f_\theta(x, t) \,\mathrm{d}t}_\text{drift} + \underbrace{L_\theta(x, t) \, \mathrm{d}\beta_t}_\text{diffusion},
$$

or as a discrete-time stochastic process,

$$
    x_t \sim \underbrace{p_\theta(x_t \,|\, x_{t-1})}_{\text{transition density}}.
$$

In either case, the dynamics are specified with a parameter $\theta$, as well as the parameters of the Brownian motion $\beta_t$ and the initial state distrbution $p(x_0)$. Various names for the process $x_t \mapsto x_{t+1}$ are used, including *transition model*, *dynamics*, *state transition*, and *transition kernel*, but we will collectively call this our *state evolution*.

We also assume that observations are also generated probabilistically -- depending only on the state \(x_t\) -- via some process

$$
    y_t \,|\, x_t \sim p_\varphi(y_t \,|\, x_t).
$$

parameterized by $\varphi$. All our statistical problems arise from a set of observations $y_{1:T} \triangleq y_1, \dots, y_T$, observed at times $t_1, \dots, t_T$. The process $p_\varphi(\cdot \,|\, \cdot)$ will be called the *observation model*, and $y_{1:T}$ *observations*, though they are sometimes called *measurements*, *emissions*, or *outputs* in the literature.

### Specification of a Dynamical System in `dynestyx`

Specifying the dynamical model in `dynestyx` follows a simple, unified interface. To fully specify a dynamical model, we require the following:

- the initial conditions, $p(x_0)$;
- the state evolution, as a `ContinuousTimeStateEvolution` or a `DiscreteTimeStateEvolution`; and
- an observation model, $p_\varphi(y_t \,|\, x_t)$.

The resulting constructor is simple, yet quite general:
```python 
from dynestyx.models import DynamicalModel, ContinuousTimeStateEvolution

dynamics = DynamicalModel(
    initial_condition=dist.MultivariateNormal(...),
    state_evolution=ContinuousTimeStateEvolution(
        drift=lambda x, u, t: ...,
        diffusion=dsx.FullDiffusion(...),
    ),
    observation_model=lambda x, u, t: ...,
)
```

To sample from the dynamical model, we call `dsx.sample` inside of any `numpyro` model:

```python
import dynestyx as dsx

dsx.sample("f", dynamics)
```

In subsequent tutorials, we give concrete examples of defining many different types of dynamical models in `dynestyx`. Some examples include:

- A linear, Gaussian SDE-driven dynamical system ([Quickstart tutorial](tutorials/quickstart.ipynb)).
- A discrete-time dynamical system with discrete states $x_t \in \{0, 1, \dots, K\}$, i.e., a Hidden Markov Model (HMM) ([HMM tutorial](tutorials/gentle_intro/07_hmm.ipynb)).
- A linear SDE with non-Gaussian (Poisson) observations ([Non-Gaussian observations tutorial](tutorials/sde_non_gaussian_observations.ipynb)).

### Simulation of Dynamical Systems

Once a dynamical model is specified, we still require ways to *simulate* from it, i.e., to sample from the state evolution $x_t \mapsto x_{t+1}$. This is particularly the case for SDEs, where exact inference is intractable, and we must rely on numerical approximation.

There are two entry points:

- [`dsx.Simulator()`](api_reference/public/simulators/simulator_wrapper.md) is a
  handler for models that use `dsx.sample(...)` inside a NumPyro or
  `Predictive` workflow.
- [`dsx.simulate(...)`](api_reference/public/handlers.md) performs standalone
  pure-JAX generation from an already constructed `DynamicalModel`.

Both inspect the model and automatically select discrete-time, ODE, or SDE
simulation. For a differential equation, pass the corresponding
[`dsx.ODESimulatorConfig`](api_reference/public/simulators/simulator_configs.md)
or
[`dsx.SDESimulatorConfig`](api_reference/public/simulators/simulator_configs.md)
through `simulator_config` to control the numerical solver. Discrete-time
models do not require a simulator configuration. In every case,
`n_simulations` specifies how many independent trajectories to generate.

For example, this draws ten SDE trajectories for each NumPyro model execution:

```python
import jax.numpy as jnp
import jax.random as jr
from numpyro.infer import Predictive

predict_times = jnp.arange(0.0, 1.0, 0.1)
simulator_config = dsx.SDESimulatorConfig(source="em_scan", dt0=1e-3)

with dsx.Simulator(
    simulator_config=simulator_config,
    n_simulations=10,
):
    samples = Predictive(
        continuous_discrete_model,
        num_samples=1,
        exclude_deterministic=False,
    )(jr.PRNGKey(0), predict_times=predict_times)
```

The same `dsx.Simulator()` call auto-routes to the discrete-time or ODE backend
when the model changes. If you do not need NumPyro sites, use the standalone
API:

```python
simulation_result = dsx.simulate(
    dynamics,
    rng_key=jr.PRNGKey(0),
    predict_times=predict_times,
    n_simulations=10,
    simulator_config=simulator_config,
)
```

Both interfaces generate states and observations at `predict_times`: use
`dsx.Simulator()` with NumPyro and `dsx.simulate(...)` otherwise. Next, we turn
to inference methods that exploit the model's temporal structure.

## Bayesian Inference of Dynamical Systems

In Bayesian inference of dynamical systems, we place priors on $\theta$ and $\varphi$, $p(\theta)$ and $p(\varphi)$, seeking to answer two different types of problems using the observed $y_{1:T}$.

#### State Inference

The first type of problem we may ask surround various posterior probabilities over the state $x_t$ conditional on one set of fixed parameters $\theta$ and $\varphi$. This is the canonical problem of Bayesian filtering and smoothing [1]. The three common problems in state estimation include

1. Filtering, where we aim to build a posterior of the current state $x_t$ given all observations up to the current time, i.e., $p(x_t \,|\, y_{1:t})$.
2. Smoothing, where we aim to build a posterior over all states $x_t$ up to an including the current time, i.e., $p(x_t \,|\, y_{1:T})$, with $T \geq t$.
3. Forecasting, where we aim to build a posterior predictive over a future state $x_{t'}$, i.e., $p(x_{t'} \,|\, y_{1:T})$, with $t' > T$.

It turns out that methods designed for each of these problems are typically well-connected theoretically. In the case of filtering and smoothing, we also get estimates of the *marginal likelihood* $p(y_{1:T} \,|\, \theta, \varphi)$, which will prove useful in our second type of problem.

#### System Identification

The second type of problem we may ask are deriving posterior distributions of $\theta$, and $\varphi$, so that our system is well-specified. This problem is called *system identification* or *parameter inference*; via Bayes' rule, we seek to estimate

$$
    p(\theta, \varphi \,|\, y_{1:T}) \propto p(\theta, \varphi) \underbrace{p(y_{1:T} \,|\, \theta, \varphi)}_{\text{marginal likelihood}}.
$$

The resulting inference problem can be extremely difficult, and is often intractable to solve naively, i.e., using standard inference methods like NUTS/HMC. This is primarily because the *marginal likelihood* $p(y_{1:T} \,|\, \theta, \varphi)$ is difficult to compute without specialized algorithms, like the filtering and smoothing algorithms above. As a result, we must use more sophisticated inference methods that combine state estimation and parameter inference.

One goal in `dynestyx` is to make the resulting inference problem as simple as possible to specify, leveraging filtering and smoothing algorithms to efficiently compute or approximate the marginal likelihood. Various examples of inference methods include:

- Hamiltonian Monte Carlo/NUTS using an EnKF to approximate the marginal likelihood (biased, but fast and generally accurate)
- Particle Hamiltonian Monte Carlo (unbiased, but slow for nontrivial problems)
- Stochastic Variational Inference using the EnKF or particle filter for marginal likelihood estimates (approximate, but extremely fast)

For concrete examples of these methods, please see the [tutorials](tutorials/gentle_intro/00_index.ipynb).

## Tutorials

A comprehensive collection of tutorials is available to help you get started with `dynestyx` and learn different inference methods for dynamical systems. The tutorials range from introductory material to advanced techniques for handling complex observation models and different types of dynamics. Visit the [tutorials page](tutorials/gentle_intro/00_index.ipynb) to explore all available guides.

## API Reference

Detailed API documentation is available for all modules, classes, and functions in `dynestyx`. The API reference provides comprehensive information about function signatures, parameters, return values, and usage examples. Visit the [API reference page](api_reference/index.md) to browse the complete documentation.

## References

[1] Särkkä, S., & Svensson, L. (2023). Bayesian Filtering and Smoothing (Vol. 17). Cambridge University Press. [[Available Online]](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf)

[^1]: We also allow for some more general state space models than are described here. For example, the state need not be real (see, for example, the Hidden Markov Model tutorials).

[^2]: As another example of support for general models, we also allow for deterministic dynamical systems (i.e., ODEs).
