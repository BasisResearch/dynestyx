# Examples

Welcome to the `dynestyx` examples page

## Getting Started

- [Quickstart](tutorials/quickstart.ipynb) - Minimal introduction to building models in `dynestyx` and performing inference
- [A Gentle Introduction to Dynestyx](tutorials/gentle_intro/00_index.ipynb) - Multi-part tutorials: NumPyro and Bayesian workflow → discrete-time dynestyx → filtering and MLL → pseudomarginal inference → SVI → continuous-time (SDE, L63) → HMMs → hierarchical / mixed-effect inference with `plate`. These have tons of examples!
- [Filtering and marginal likelihood without NumPyro](tutorials/gentle_intro/03_filtering_mll_no_numpyro.ipynb) - Use `dsx.condition` to return filter results directly.
- [Differentiable optimization without NumPyro](tutorials/gentle_intro/05_svi_no_numpyro.ipynb) - Optimize filter marginal likelihoods and explicit latent-path log densities with JAX and Optax.

## Featured Examples { #gallery }

<div class="grid cards" markdown>

-   ![Latent trajectory above observed Poisson counts and their expected rate](assets/gallery/sde_non_gaussian_observations.png)

    **[SDEs with non-Gaussian observations](tutorials/sde_non_gaussian_observations.ipynb)**

    Fit a continuous-time latent process to Poisson count observations with variational inference.

    <!-- Thumbnail source: notebook cell 7, output 0 (zero-based indices). Trimmed and padded. -->

-   ![Lorenz–63 posterior densities with true rho marked and the Euler–Maruyama peak far to its left](assets/gallery/sde_discretization_comparison.png)

    **[Comparing SDE discretizations on Lorenz–63](deep_dives/sde_discretization_comparison.ipynb)**

    See how the discretization method changes parameter recovery and sampling efficiency.

    <!-- Thumbnail source: cell posterior-figure, output 0. Removed the original legend and labeled the Euler–Maruyama peak and true rho directly. -->

-   ![Gaussian log score versus inflation with the best grid point and optimizer result near the peak](assets/gallery/l63_covariance_inflation_scoring.png)

    **[Tuning EnKF covariance inflation with scoring rules](deep_dives/l63_covariance_inflation_scoring.ipynb)**

    Tune ensemble covariance inflation for Lorenz–63 by scoring predictions on held-out observations.

    <!-- Thumbnail source: cell 10e1febf, output 0. Trimmed and padded. -->

-   ![Observed and latent Lorenz–96 sites beside an illustrative taper that gives nearby state–observation pairs greater weight](assets/gallery/l96_localization_hyperparameter_scoring.png)

    **[Learning EnKF localization with scoring rules](deep_dives/l96_localization_hyperparameter_scoring.ipynb)**

    Use predictive scores to select how far an observation can influence nearby states on the Lorenz–96 ring.

    <!-- Thumbnail source: cell 075d6d45, output 0. Retained the ring, state–observation taper and shared scale at illustrative length 2.5. -->

-   ![Seven samplers ranked by effective samples per second, with adaptive Metropolis-within-Gibbs first in this run](assets/gallery/mcmc_inference_algorithm_comparison.png)

    **[Comparing MCMC algorithms](deep_dives/mcmc_inference_algorithm_comparison.ipynb)**

    Compare posterior estimates and sampling efficiency across NumPyro and BlackJAX samplers.

    <!-- Thumbnail source: cell b1f40fe8, output 11. Replotted all seven saved ESS/s values on a logarithmic axis. -->

-   ![Gradient matching takes 4.65 seconds and EnKF takes 116.35 seconds in one full-state, low-noise run](assets/gallery/l63_speedup_dirac_vs_enkf.png)

    **[Faster Lorenz–63 inference with full-state observations](deep_dives/l63_speedup_dirac_vs_enkf.ipynb)**

    Compare gradient matching and ensemble Kalman filtering when every state coordinate is observed with little noise.

    <!-- Thumbnail source: cells 8 and 10, output 1 in each (zero-based indices). Replotted the two reported runtimes as bars. -->

-   ![True and inferred FitzHugh–Nagumo coefficients show similar patterns of nonzero terms](assets/gallery/fhn_sparse_id.png)

    **[Sparse identification of FitzHugh–Nagumo dynamics](deep_dives/fhn_sparse_id.ipynb)**

    Recover the active polynomial terms of a nonlinear dynamical system from noisy observations.

    <!-- Thumbnail source: notebook cell 25, output 0 (zero-based indices). Retained both heatmaps and their common color scale. -->

-   ![Exact Kalman and stochastic filter likelihood profiles peak near the true continuous-time drift parameter](assets/gallery/continuous_time_lti_profile_likelihood.png)

    **[Continuous-time LTI profile likelihood](deep_dives/continuous_time_lti_profile_likelihood.ipynb)**

    Compare approximate filter likelihoods against the exact continuous-discrete Kalman filter.

    <!-- Thumbnail source: cell 7519c00c, output 0. Retained the saved profiles and shortened the legend to the exact CD-KF, stochastic CD-DPF, and true alpha. -->

-   ![Split posterior clusters for two full-model coefficients beside overlapping densities for one canonical coefficient](assets/gallery/lti_identifiability_canonicalization.png)

    **[LTI identifiability and canonical parameterization](deep_dives/lti_identifiability_canonicalization.ipynb)**

    See how equivalent state-space models can have different parameters and compare their posteriors in canonical coordinates.

    <!-- Thumbnail sources: cells bbb5a659 and 6075ca12, outputs 2 and 0 respectively. Retained the leftmost panel from each. -->

-   ![True and fitted maps of the first FitzHugh–Nagumo drift component on a shared color scale](assets/gallery/gp_drift.png)

    **[Learning FitzHugh–Nagumo drift with a Gaussian process](deep_dives/gp_drift.ipynb)**

    Use a Gaussian process prior to learn a nonlinear drift field from noisy trajectories.

    <!-- Thumbnail source: notebook cell 20, output 1 (zero-based indices). Retained the first true and MAP drift panels and their shared color scale. -->

</div>

## All Examples and Deep Dives

### Examples

<div class="grid cards" markdown>

-   ![Three simulated Lorenz–63 state coordinates above a noisy view of the first coordinate](assets/gallery/quick_example.png)

    **[Quick example: Lorenz–63](quick_example.ipynb)**

    <!-- Thumbnail source: cell 03a43c82, output 0. Trimmed and padded. -->

-   ![Latent trajectory above observed Poisson counts and their expected rate](assets/gallery/sde_non_gaussian_observations.png)

    **[SDEs with non-Gaussian observations](tutorials/sde_non_gaussian_observations.ipynb)**

</div>

#### Closed-loop control

<div class="grid cards" markdown>

-   ![Closed-loop states approach zero under feedback, with control inputs shown below](assets/gallery/controller_demo.png)

    **[Controller basics](tutorials/control/controller_demo.ipynb)**

    <!-- Thumbnail source: cell 538926f3, output 0. Trimmed and padded. -->

-   ![State norms and controller inputs for a two-dimensional system with and without optimized feedback](assets/gallery/control_optimization.png)

    **[Optimizing a controller](tutorials/control/control_optimization.ipynb)**

    <!-- Thumbnail source: cell 5d6197c9, output 0. Retained the state norm and control input panels. -->

-   ![Observed and hidden state coordinates under MPPI and without control, with MPPI inputs below](assets/gallery/mpc_demo.png)

    **[MPPI-inspired model predictive control](tutorials/control/mpc_demo.ipynb)**

    <!-- Thumbnail source: cell 5dbcf7d2, output 0. Trimmed and padded. -->

</div>

#### State-space models

<div class="grid cards" markdown>

-   ![Kalman-filtered object path and uncertainty ellipses alongside noisy position measurements](assets/gallery/kf_tracking.png)

    **[Object tracking with a Kalman filter](tutorials/state_space_models/kf_tracking.ipynb)**

    <!-- Thumbnail source: cell 2cbb6d25, output 0. Trimmed and padded. -->

-   ![Online estimates of two regression weights approach the offline estimates over time](assets/gallery/kf_linreg.png)

    **[Online linear regression with a Kalman filter](tutorials/state_space_models/kf_linreg.ipynb)**

    <!-- Thumbnail source: cell 1bd0dad6, output 0. Trimmed and padded. -->

-   ![NUTS posterior histogram for a model parameter with the true value and SGD estimate marked](assets/gallery/lgssm_learning.png)

    **[Parameter estimation for linear Gaussian state-space models](tutorials/state_space_models/lgssm_learning.ipynb)**

    <!-- Thumbnail source: cell 43c86dff, output 0. Trimmed and padded. -->

-   ![Extended Kalman filter estimates and uncertainty ellipses follow a curling trajectory through noisy measurements](assets/gallery/ekf_spiral.png)

    **[Tracking a spiraling object with an extended Kalman filter](tutorials/state_space_models/ekf_spiral.ipynb)**

    <!-- Thumbnail source: cell fca4aee3, output 0. Trimmed and padded. -->

-   ![Neural-network posterior predictions follow the target curve after seeing 200 observations](assets/gallery/ekf_mlp.png)

    **[Online learning of a neural network with an extended Kalman filter](tutorials/state_space_models/ekf_mlp.ipynb)**

    <!-- Thumbnail source: cell ee4d1279, output 3. Trimmed and padded. -->

</div>

### Deep Dives

#### Filtering and predictive scoring

<div class="grid cards" markdown>

-   ![Runtime and speedup curves compare sequential and associative Kalman filtering across sequence lengths](assets/gallery/discrete_time_kf_associative_scaling.png)

    **[Scaling the associative Cuthbert Kalman filter](deep_dives/discrete_time_kf_associative_scaling.ipynb)**

    <!-- Thumbnail source: cell plot-results, output 0. Trimmed and padded. -->

-   ![Grouped bars compare state-recovery error and 90 percent interval coverage across filters](assets/gallery/filtering_deterministic_processes.png)

    **[Filtering deterministic processes](deep_dives/filtering_deterministic_processes.ipynb)**

    <!-- Thumbnail source: cell summarize-state-recovery, output 1. Trimmed and padded. -->

-   ![Four time-series panels compare predictive scores for two observation models](assets/gallery/observation_scoring_with_cuthbert_enkf.png)

    **[Observation scoring with a Cuthbert EnKF](deep_dives/observation_scoring_with_cuthbert_enkf.ipynb)**

    <!-- Thumbnail source: cell 3aeb0829, output 1. Trimmed and padded. -->

-   ![Gaussian log score versus inflation with the best grid point and optimizer result near the peak](assets/gallery/l63_covariance_inflation_scoring.png)

    **[Tuning EnKF covariance inflation with scoring rules](deep_dives/l63_covariance_inflation_scoring.ipynb)**

-   ![Observed and latent Lorenz–96 sites beside an illustrative taper that gives nearby state–observation pairs greater weight](assets/gallery/l96_localization_hyperparameter_scoring.png)

    **[Learning EnKF localization with scoring rules](deep_dives/l96_localization_hyperparameter_scoring.ipynb)**

</div>

#### Parameter inference and identifiability

<div class="grid cards" markdown>

-   ![Gradient matching takes 4.65 seconds and EnKF takes 116.35 seconds in one full-state, low-noise run](assets/gallery/l63_speedup_dirac_vs_enkf.png)

    **[Faster Lorenz–63 inference with full-state observations](deep_dives/l63_speedup_dirac_vs_enkf.ipynb)**

-   ![Seven samplers ranked by effective samples per second, with adaptive Metropolis-within-Gibbs first in this run](assets/gallery/mcmc_inference_algorithm_comparison.png)

    **[Comparing MCMC algorithms](deep_dives/mcmc_inference_algorithm_comparison.ipynb)**

-   ![Lorenz–63 posterior densities with true rho marked and the Euler–Maruyama peak far to its left](assets/gallery/sde_discretization_comparison.png)

    **[Comparing SDE discretizations on Lorenz–63](deep_dives/sde_discretization_comparison.ipynb)**

-   ![Discrete-time parameter likelihood profiles from several filters with the simulated parameter marked](assets/gallery/discrete_time_lti_profile_likelihood.png)

    **[Discrete-time LTI profile likelihood](deep_dives/discrete_time_lti_profile_likelihood.ipynb)**

    <!-- Thumbnail source: notebook cell 15 (zero-based), output 0. Trimmed and padded. -->

-   ![Exact Kalman and stochastic filter likelihood profiles peak near the true continuous-time drift parameter](assets/gallery/continuous_time_lti_profile_likelihood.png)

    **[Continuous-time LTI profile likelihood](deep_dives/continuous_time_lti_profile_likelihood.ipynb)**

-   ![Split posterior clusters for two full-model coefficients beside overlapping densities for one canonical coefficient](assets/gallery/lti_identifiability_canonicalization.png)

    **[LTI identifiability and canonical parameterization](deep_dives/lti_identifiability_canonicalization.ipynb)**

</div>

#### Learning dynamics

<div class="grid cards" markdown>

-   ![True and inferred FitzHugh–Nagumo coefficients show similar patterns of nonzero terms](assets/gallery/fhn_sparse_id.png)

    **[Sparse identification of FitzHugh–Nagumo dynamics](deep_dives/fhn_sparse_id.ipynb)**

-   ![Predator and prey phase-space loop with the true trajectory, noisy observations, and inferred uncertainty](assets/gallery/lv_uode.png)

    **[Learning Lotka–Volterra dynamics with a universal ODE](deep_dives/lv_uode.ipynb)**

    <!-- Thumbnail source: cell d5fa65a1, output 1. Trimmed and padded. -->

-   ![True and fitted maps of the first FitzHugh–Nagumo drift component on a shared color scale](assets/gallery/gp_drift.png)

    **[Learning FitzHugh–Nagumo drift with a Gaussian process](deep_dives/gp_drift.ipynb)**

</div>

#### Partial differential equations

<div class="grid cards" markdown>

-   ![Space-time heatmap showing an initial temperature peak spreading and fading](assets/gallery/heat_equation_inference.png)

    **[Heat equation: Inferring an initial condition](deep_dives/pde/heat_equation_inference.ipynb)**

    <!-- Thumbnail source: cell a44a243e, output 0. Trimmed and padded. -->

-   ![Space-time heatmap of the KPP reaction–diffusion solution reconstructed at an inferred rate](assets/gallery/reaction_diffusion_inference.png)

    **[Reaction–diffusion equation: Inferring a reaction rate](deep_dives/pde/reaction_diffusion_inference.ipynb)**

    <!-- Thumbnail source: cell ffca77e2, output 0. Trimmed and padded. -->

</div>
