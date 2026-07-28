# Examples

Welcome to the `dynestyx` examples page

## Getting Started

- [Quickstart](tutorials/quickstart.ipynb) - Minimal introduction to building models in `dynestyx` and performing inference
- [A Gentle Introduction to Dynestyx](tutorials/gentle_intro/00_index.ipynb) - Multi-part tutorials: NumPyro and Bayesian workflow → discrete-time dynestyx → filtering and MLL → pseudomarginal inference → SVI → continuous-time (SDE, L63) → HMMs → hierarchical / mixed-effect inference with `plate`. These have tons of examples!
- [Filtering and marginal likelihood without NumPyro](tutorials/gentle_intro/03_filtering_mll_no_numpyro.ipynb) - Use `dsx.condition` to return filter results directly.
- [Differentiable optimization without NumPyro](tutorials/gentle_intro/05_svi_no_numpyro.ipynb) - Optimize filter marginal likelihoods and explicit latent-path log densities with JAX and Optax.

## Gallery

- [SDEs with Non-Gaussian Observations](tutorials/sde_non_gaussian_observations.ipynb)
- [Comparing Different MCMC Algorithms](deep_dives/mcmc_inference_algorithm_comparison.ipynb)
- [HUGE speedups if you assume perfect observations](deep_dives/l63_speedup_dirac_vs_enkf.ipynb)
- [SINDy (Sparse identification of non-linear dynamics)](deep_dives/fhn_sparse_id.ipynb)
- [Continuous-time LTI profile likelihood](deep_dives/continuous_time_lti_profile_likelihood.ipynb)
- [GP prior on drift: learning FitzHugh–Nagumo dynamics](deep_dives/gp_drift.ipynb)
- Using `Discretizer` to go from continuous-time -> discrete - Coming soon!
