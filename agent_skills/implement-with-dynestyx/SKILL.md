---
name: implement-with-dynestyx
description: Re-implements an existing piece of work, usually a research paper, as a minimal and tested dynestyx notebook or sub-repo. Use when the user wants to reproduce or port a dynamical-systems or state-space paper into dynestyx, points at a paper PDF or an arXiv or biorxiv link and asks for a dynestyx version, or asks to build a state-space, filtering, particle-filter, smoothing, pseudo-marginal MCMC, or SVI model in dynestyx from a described model. Use this even when the user does not name the skill but describes a dynamical-systems modeling task to carry out in dynestyx.
---

# Implement with dynestyx

`dynestyx` is a NumPyro extension for Bayesian inference in dynamical systems. A model is a NumPyro model function that builds a `DynamicalModel` and returns `dsx.sample(...)`. Inference is selected by wrapping that same model in a handler (`Filter`, `Smoother`, or a simulator) and running it through NumPyro's `Predictive`, `MCMC`, or `SVI`. The same model serves every inference method.

## Workflow

Copy this checklist into your response and tick items off as you go.

```
- [ ] 1. Acquire the source
- [ ] 2. Extract the model spec
- [ ] 3. Grill the user on requirements
- [ ] 4. Map the spec to dynestyx
- [ ] 5. Build, validate, deliver
```

### 1. Acquire the source

If the input is a PDF, first ask the user whether they can provide the TeX source. TeX gives exact equations, while a PDF forces error-prone extraction of math. If the user cannot provide it, read the PDF and find the source online, and fall back to the PDF when no source exists. If the paper ships code, pull that codebase into a scratch location and keep it as a reference for the dynamics and the data, without letting its style drive yours.

### 2. Extract the model spec

Distill a precise spec that the later phases can implement directly. A complete spec states the latent state and its dimension, the dynamics and whether they are discrete or continuous in time and deterministic or stochastic, the observation model, the parameters and which of them are unknown, the priors on the unknowns, and the inference target of states or parameters or both. The discrete-versus-continuous and linear-Gaussian-versus-nonlinear choices decide most of the downstream API, so settle them here.

### 3. Grill the user on requirements

Invoke the `grill-with-docs` skill to settle the open requirements with the user. Cover the deliverable shape of a single notebook versus a sub-repo and where it lands, whether to use simulated data or the paper's real data, which model hypotheses from the paper to include, and the concrete success signal that will count as done.

### 4. Map the spec to dynestyx

Read [reference.md](reference.md), then exactly one inference file picked from the spec. Translate the spec into idiomatic dynestyx using only those files. Avoid reading the wider `docs/` tree.

- Online state estimates or the marginal likelihood, via Kalman-family, particle, or HMM filters → [inference-filtering.md](inference-filtering.md)
- State estimates that use the full observation window → [inference-smoothing.md](inference-smoothing.md)
- Posterior over unknown parameters, pseudo-marginal or joint MCMC → [inference-mcmc.md](inference-mcmc.md)
- Fast approximate posterior for larger problems → [inference-svi.md](inference-svi.md)

### 5. Build, validate, deliver

Write the deliverable and run it end to end. Show one correctness signal, such as recovering known generating parameters on simulated data or reproducing the paper's qualitative result. Each inference file names the natural signal for its method. Do not hand over code that has not been executed.
