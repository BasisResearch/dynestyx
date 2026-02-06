# Welcome to Dynestyx

![dynestyx logo](docs/logo/dynestyx.gif)

`dynestyx` is a library designed for Bayesian modeling and inference of dynamical systems. It is an extension of [NumPyro](https://num.pyro.ai/en/stable/), and incorporates a wide variety of state-of-the-art inference methods for state space models.

## Goals of `dynestyx`

The goal of `dynestyx` is to decouple model code and inference code for dynamical systems, a common theme in *probabilistic programming languages* like [NumPyro](https://num.pyro.ai/en/stable/). The benefits of this are two-fold: modellers get an interface that is simple to use, with access to advanced inference methods for free. Methods researchers simultaneously get a platform where their methodologies can be immediately used, with a natural testbed of problems to evaluate performance on.

### Relation to Existing Libraries

While many probabilistic programming languages now exist (e.g., [Pyro](https://pyro.ai/), [NumPyro](https://num.pyro.ai/en/stable/), and [Stan](https://mc-stan.org/)), these solutions do not offer support of structured inference methods specifically designed for the dynamical systems setting, leading to subpar inference and ad-hoc code that may be difficult to write for practitioners. In `dynestyx`, we treat dynamical systems as first-class objects, with direct interfacing to methods like pseudo-marginal MCMC and stochastic variational inference for parameter inference.

Simultaneously, many strong solutions exist for inference in dynamical systems; modern examples include [dynamax](https://github.com/probml/dynamax) for discrete-time dynamical systems, [cd-dynamax](https://github.com/hd-UQ/cd_dynamax) for continuous-time dynamical systems, and [PFJax](https://pfjax.readthedocs.io/en/latest/) for nonlinear and non-Gaussian discrete-time dynamical systems. While featureful, one drawback of this suite of methods is a varied set of APIs, with model code that is tightly coupled with the resulting inference method. In `dynestyx`, we offer a large variety of different inference methods under the same roof in a unified, abstract API. Iterating and selecting the appropriate inference methods is thus a significantly simpler process. Using tools from PPLs, we are also able to introspectively analyze a given model, and select appropriate inference methods which take advantage of model structure (e.g., linearity or Gaussianity).

## Installation

For installation, we recommend [`uv`](https://docs.astral.sh/uv/):
```bash
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/BasisResearch/dynestyx.git
```

But `pip` works as well:
```bash
pip install git+https://github.com/BasisResearch/dynestyx.git
```

## Quickstart

We provide a more mathematical introduction in the [Introduction](intro.md) section. For a hands-on tutorial with code examples, check out the [Quickstart Tutorial](tutorials/quickstart/). 