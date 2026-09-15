# Introducing Dynestyx, a probabilistic programming library for dynamical systems

One of our primary goals at Basis is to identify and address unmet computational needs that permeate across sectors of science and engineering. It allows us to play a uniquely catalyzing role in research that most institutions are unable to incentivize. Roughly one year ago, we set out on one such mission, with a laser focus on improving how applied and methodological researchers/scientists/engineers work with data and models from time-evolving (dynamical) systems to answer key questions like:

1. What will happen next in this system? (*forecasting*)
2. Where has the system been? (*inference*, generically, or *smoothing* in certain sub-communities)
3. How does the system work / what are its rules? (*system identification*)
4. What will happen if I intervene on the system (*counterfactual inference*)?

Good answers to these questions are paired with quantified uncertainty—that is, how sure are we about the answer? The best answers also address the meta-question: how reliable is the assessment of uncertainty? During the second wave of COVID-19 in Germany and Poland in late 2020, different forecasting teams often produced 95% prediction intervals with little or no overlap. Most models’ intervals also contained the eventual outcomes less often than their stated coverage levels would suggest—evidence that they underestimated forecast uncertainty.[^covid-forecasts]

Countless fields rely on high-quality answers to these questions, from epidemiology to robotics engineering, and from atmospheric sciences to biomedical device engineering. Supporting these applied needs is a large and robust community of methodological researchers in fields that include applied mathematics, statistics, computer science, and engineering.

However, the interactions within and between these applied and methodological fields alike leave much to be desired; in particular, a suite of reliable, efficient, and general purpose tools that can serve the majority of evolving needs across these fields.

## Our assessment.

There is a critical lack of reliable, efficient, and general purpose tools that can serve the majority of evolving needs across these fields. It is common for a methodological research group to maintain a Python package that deploys their latest or most used algorithms. However, there are hundreds of these that live in isolation (each with their own scope), creating frustrating incompatibilities in software, mathematical underpinnings, and data particularities. In applied domains, it is common for a field to quickly converge on a specific tool with a (sometimes modest) success history. It is also common for individual groups to re-implement generic toolings in a field-specific lens.

The result is dozens of brittle echo-chambers in which comparing methods within a given problem is unnecessarily difficult and where methodologists are needlessly disconnected from challenges specific to particular applied domains.

A great example I came across recently is this paper[^fish-telemetry] which uses `patter`[^patter], an R and Julia package for tracking animal movement from acoustic telemetry data; it makes some specific modeling choices that are highly relevant to the setting, but the algorithm itself (while written nicely) is quite standard (and, in fact, does not take advantage of recent advances in SOTA PFs). With the right tooling in place, one might view this repository similarly to one with a custom implementation of linear regression specially designed for measurements of the microbiome. To be clear, I think this work is great—I really like that paper! But what happens as state-of-the-art filtering improves? Is it really the responsibility of ecological scientists to stay up to date on the latest and greatest in Particle Filtering? I hope not! If only there were a general tool maintained and updated by a collective community of methodological and applied researchers that ensures cutting-edge, generalist support for the most commonly used algorithms and practices in dynamical systems modeling and inference?!

## This is why we built Dynestyx[^dynestyx-preprint].

We introduce `dynestyx`, a probabilistic programming library that treats dynamical systems as first-class objects. `dynestyx` builds on top of `numpyro` to provide a clean, unified interface for Bayesian state-space models, providing a one-stop-shop for parameter inference and state inference.

[GitHub](https://github.com/BasisResearch/dynestyx) · [Docs](https://basisresearch.github.io/dynestyx/stable/) · [Preprint](https://arxiv.org/abs/2606.16985)

Our goal in building `dynestyx` is two-fold, serving both practitioners and theoreticians. On the side of application, `dynestyx` provides an approachable interface for model-building, remaining expressive over model choices and providing a variety of state-of-the-art inference methods. On the side of methodology, `dynestyx` offers a natural integration surface to implement, test, and apply new algorithmic approaches in real-world problems. This is reminiscent of `stan`[^stan], which has spurred both applied work in Bayesian statistics and methodological work in Monte Carlo and variational inference.

## A unified interface

Core to `dynestyx` is its unified interface for a state-space model (SSM, defined below). In particular, `dynestyx` supports inference across a wide variety of model classes belonging to the SSM family. A core thesis of ours is that dynestyx should look a lot like the math you use to describe your model, so let's start there.

### Mathematical description of an SSM

A state-space model concerns the evolution of a *latent state* $x_t \in\mathbb{R}^{d_x}$, beginning with a possibly uncertain initial condition:
$$x_0 \sim \pi_0$$
The subsequent evolution may be deterministic or stochastic, occur continuously or discretely in time, and may have explicit dependence on time $t \in \mathbb{R}^+$ or a sequence of external control inputs $u_t \in \mathbb{R}^{d_u}$. Let's consider the stochastic case for now, since it's more general. In discrete-time, our state then evolves according to a Markov chain with some *rules* $\theta$ :

$$
x_t \sim p(x_t | x_{t-1}, u_{t-1}, t-1, t; \ \theta).
$$

In continuous time, we instead have a stochastic differential equation (SDE):

$$
\mathrm{d}x_t = f(x_t, u_t, t)  \mathrm{d}t + g(x_t, u_t, t)  \mathrm{d}W_t,
$$

where $W_t$ is a $d_b$-dimensional Brownian motion, $`f \colon \mathbb{R}^{d_x} \times \mathbb{R}^{d_u} \times \mathbb{R}^{+} \to \mathbb{R}^{d_x}`$ is the drift function (governing deterministic dynamics), and $`g \colon \mathbb{R}^{d_x} \times \mathbb{R}^{d_u} \times \mathbb{R}^{+} \to \mathbb{R}^{d_x \times d_b}`$ is the diffusion coefficient function (governing the coupling to the stochastic part of the evolution).

In either case, an observation model connects the latent state to the data we measure at observation times $t_k$:

$$
y_k \sim p(y_k \mid x_{t_k}, u_{t_k}, t_k; \ \theta).
$$

Here, $y_k$ is the observed data, and the observation model describes how those measurements depend on the latent state, including measurement noise. The observation times may be irregularly spaced.

### Mathematics to Code

Given the mathematical description of a state-space model, it is straightforward to translate to a `dynestyx` model! The key abstraction in `dynestyx` is a `DynamicalModel`, which takes as input exactly the data we specified above: an initial condition (`initial_condition`), a state evolution (`state_evolution`), and an observation model (`observation_model`):

|                 | stochastic                                                                                                                | deterministic                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| continuous-time | $\mathrm{d}x_t = f(x_t, u_t, t)   \mathrm{d}t + g(x_t, u_t, t)   \mathrm{d}W_t$<br><br>`ContinuousTimeStateEvolution(f, g)` | $\mathrm{d}x_t = f(x_t, u_t, t)   \mathrm{d}t$<br><br>`ContinuousTimeStateEvolution(f)`               |
| discrete-time   | $x_t \sim p(x_t \mid x_{t-1}, u_{t-1}, t-1, t;\ \theta)$<br><br>`DiscreteTimeStateEvolution(p)`                           | $x_t = f(x_{t-1}, u_{t-1}, t-1, t;\ \theta)$<br><br>`DiscreteTimeStateEvolution(DiracTransition(f))` |

## Let's see Dynestyx in action!

### Missing observations in hidden Markov models

![Hidden-state probabilities and categorical sensor observations, including missing measurements](figures/tutorials/gentle_intro/11c_missing_observations_hmms/figure-06_cell-010_output-04.png)

Infer hidden states from two categorical sensors even when individual measurements or entire stretches of observations are missing. [Explore the notebook](../docs/tutorials/gentle_intro/11c_missing_observations_hmms.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
posterior_cat, A_post_mean_cat, filtered_cat = infer_hmm(
    independent_cat_hmm,
    OBS_TIMES,
    obs_missing_cat,
    fit_key=FIT_KEY_CAT,
    recon_key=RECON_KEY_CAT,
)
```

</details>

### Learning parameters and trajectories with missing data

<table>
  <tr><td><img src="figures/tutorials/gentle_intro/11b_missing_observations_latent_path_mcmc/figure-03_cell-008_output-01.png" alt="Posterior distributions for alpha from latent-path MCMC and Kalman smoothing" width="900"></td></tr>
  <tr><td><img src="figures/tutorials/gentle_intro/11b_missing_observations_latent_path_mcmc/figure-04_cell-008_output-03.png" alt="Paired latent-state reconstructions with uncertainty across a missing-data interval" width="900"></td></tr>
</table>

Compare posterior distributions for the dynamics parameter alpha and reconstructed trajectories using latent-path MCMC and a Kalman smoother. The trajectory bands widen across the missing-data interval. [Explore the notebook](../docs/tutorials/gentle_intro/11b_missing_observations_latent_path_mcmc.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
def conditioned_latent_path(obs_times=None, obs_values=None):
    with dsx.LatentPathBuilder():
        ar1_model(obs_times=obs_times, obs_values=obs_values)


def conditioned_smoother(obs_times=None, obs_values=None):
    with Smoother(
        smoother_config=KFSmootherConfig(
            filter_source="cuthbert",
            record_smoothed_states_mean=True,
            record_smoothed_states_cov_diag=True,
        )
    ):
        ar1_model(obs_times=obs_times, obs_values=obs_values)
```

</details>

### Smoothing the past, simulating the future

![Smoothed continuous-time states followed by future simulations and 90 percent rollout intervals](figures/tutorials/gentle_intro/10_continuous_smoothing/figure-05_cell-023_output-01.png)

Combine a continuous-time smoother with a simulator to reconstruct partially observed states and generate future trajectories with 90% rollout intervals. [Explore the notebook](../docs/tutorials/gentle_intro/10_continuous_smoothing.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
future_times = jnp.linspace(obs_times[-1], obs_times[-1] + 1.5, 31)

forecast_predictive = Predictive(
    continuous_lti_model,
    params={"rho": rho_post_mean},
    num_samples=1,
    exclude_deterministic=False,
)

n_rollout = 40

with Simulator(n_simulations=n_rollout):
    with Smoother(
        smoother_config=ContinuousTimeKFSmootherConfig(
            record_smoothed_states_mean=True,
            record_smoothed_states_cov_diag=True,
        )
    ):
        forecast = forecast_predictive(
            jr.PRNGKey(3),
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=future_times,
        )
```

</details>

### Learning across related trajectories

<table>
  <tr><td><img src="figures/tutorials/gentle_intro/08_hierarchical_inference/figure-03_cell-013_output-01.png" alt="Eight simulated trajectories with different equilibrium and initial-condition means" width="900"></td></tr>
  <tr><td><img src="figures/tutorials/gentle_intro/08_hierarchical_inference/figure-04_cell-018_output-01.png" alt="Posterior distributions of trajectory-specific equilibrium and initial-condition means" width="900"></td></tr>
</table>

Fit a hierarchical Ornstein–Uhlenbeck model to multiple trajectories. The paired plots show the simulated paths and posterior distributions for each trajectory’s equilibrium and initial-condition means. [Explore the notebook](../docs/tutorials/gentle_intro/08_hierarchical_inference.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
def conditioned_hierarchical_ou_model():
    with Filter(ContinuousTimeKFConfig(warn=False)):
        return hierarchical_ou_model(
            N_trajectories=N_trajectories,
            obs_times=obs_times,
            obs_values=obs_values,
        )


mcmc = MCMC(NUTS(conditioned_hierarchical_ou_model), num_warmup=100, num_samples=100)
mcmc.run(jr.PRNGKey(2))
posterior = mcmc.get_samples()
```

</details>

### Forecasting a partially observed chaotic system

![Lorenz–63 state estimates and future rollout intervals with only the first component observed](figures/tutorials/gentle_intro/06_continuous_time/figure-03_cell-015_output-02.png)

Observe only the first Lorenz–63 component, estimate all three states with an ensemble Kalman filter, and simulate future trajectories with uncertainty. [Explore the notebook](../docs/tutorials/gentle_intro/06_continuous_time.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
rho_post_mean = jnp.mean(posterior["rho"])
n_sim = 30
num_samples = 2  # Change this to 1 or >1 to test both cases

predictive = Predictive(
    l63_model,
    params={"rho": jnp.array(rho_post_mean)},
    num_samples=num_samples,
    exclude_deterministic=False,
)
with SDESimulator(
    simulator_config=dsx.SDESimulatorConfig(source="em_scan"),
    n_simulations=n_sim,
):
    with Filter(filter_config=ContinuousTimeEnKFConfig(n_particles=50, record_filtered_states_mean=True, record_filtered_states_cov_diag=True)):
        samples = predictive(
            jr.PRNGKey(99),
            obs_times=times_train_full,
            obs_values=observations_train,
            predict_times=times_test_full,
        )
```

</details>

### Learning a controller

![Initial and optimized feedback-control trajectories, state norms, and control inputs](figures/tutorials/control/control_optimization/figure-02_cell-016_output-01.png)

Differentiate through controlled dynamics to optimize a linear feedback policy. The learned controller brings the two-dimensional state toward zero more quickly than the initial policy in this rollout. [Explore the notebook](../docs/tutorials/control/control_optimization.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
predict_times_short = jnp.arange(0.0, 5.0)

def rollout_final_state_norm(K: float, key):
    policy = LinearPolicy(K=K)
    res=  dsx.simulate(
        dynamics,
        rng_key=key,
        predict_times=predict_times_short,
        control_policy=policy,
        filter_config=KFConfig(filter_source="cuthbert", record_filtered_states_mean=True),
    )
    return jnp.linalg.norm(res.states[0, -1]) # final state norm
```

</details>

### Learning unknown interactions with a universal ODE

<table>
  <tr><td><img src="figures/deep_dives/lv_uode/figure-03_cell-023_output-01.png" alt="True and inferred predator–prey interaction coefficients" width="900"></td></tr>
  <tr><td><img src="figures/deep_dives/lv_uode/figure-04_cell-024_output-01.png" alt="True and inferred process and observation noise" width="900"></td></tr>
  <tr><td><img src="figures/deep_dives/lv_uode/figure-05_cell-026_output-02.png" alt="Filtered prey and predator trajectories using inferred parameters" width="900"></td></tr>
</table>

Keep the known predator–prey growth and decay terms, and learn the unknown interactions with a sparse polynomial model. Compare recovered coefficients, noise estimates, and filtered trajectories. [Explore the notebook](../docs/deep_dives/lv_uode.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
def drift(x):
    known = lv_known_drift(x)               # (alpha*x, -delta*y)
    phi   = interaction_library(x)          # (N_TERMS,)
    unknown = Theta @ phi                   # (state_dim,)
    return known + unknown
```

</details>

### Tuning covariance inflation with proper scoring rules

<table>
  <tr><td><img src="figures/deep_dives/l63_covariance_inflation_scoring/figure-02_cell-011_output-02.png" alt="Predictive scoring-rule profiles across covariance inflation settings" width="900"></td></tr>
  <tr><td><img src="figures/deep_dives/l63_covariance_inflation_scoring/figure-04_cell-018_output-02.png" alt="Lorenz–63 state recovery with optimized and default covariance inflation" width="900"></td></tr>
</table>

Use predictive scoring rules to tune ensemble Kalman filter covariance inflation, then compare state recovery under optimized and default inflation settings. [Explore the notebook](../docs/deep_dives/l63_covariance_inflation_scoring.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
def mean_score_vector(inflation_delta):
    filter_config = make_enkf_config(inflation_delta)
    with Evaluation(observation_scoring_config=scoring_config):
        with Filter(filter_config=filter_config):
            with Discretizer(
                discretizer_config=ODEFlowConfig(ODESimulatorConfig(dt0 = FILTER_DT0))
            ):
                result = dsx.condition(
                    "f",
                    l63_dynamics(),
                    obs_times=obs_times,
                    obs_values=obs_values,
                )
    score_arrays = result.evaluation_result.observation_scores
    return jnp.stack([
        jnp.mean(score_arrays[site_name])
        for site_name, _, _ in METRIC_SPECS
    ])
```

</details>

### Discovering FitzHugh–Nagumo dynamics with Bayesian SINDy

<table>
  <tr><td><img src="figures/deep_dives/fhn_sparse_id/figure-04_cell-026_output-01.png" alt="True and inferred sparse FitzHugh–Nagumo drift coefficients" width="900"></td></tr>
  <tr><td><img src="figures/deep_dives/fhn_sparse_id/figure-05_cell-027_output-01.png" alt="True and inferred diffusion and observation noise scales" width="900"></td></tr>
  <tr><td><img src="figures/deep_dives/fhn_sparse_id/figure-06_cell-028_output-02.png" alt="FitzHugh–Nagumo phase-space reconstruction with filtered uncertainty ellipses" width="900"></td></tr>
</table>

Learn sparse polynomial drift coefficients and noise scales from noisy observations. The phase-space reconstruction shows the resulting filtered dynamics and uncertainty. [Explore the notebook](../docs/deep_dives/fhn_sparse_id.ipynb).

<details>
<summary>Show code excerpt</summary>

Excerpt from the linked notebook; imports, data, and supporting definitions are provided there.

```python
Theta = numpyro.sample(
    "Theta",
    dist.Laplace(0.0, 0.1).expand([state_dim, N_TERMS]).to_event(2),
)

sigma_x = numpyro.sample("sigma_x", dist.HalfNormal(0.1))

sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.5))

def drift(x, u, t):
    phi = monomials(x)   # phi(x) in R^{N_TERMS}
    return Theta @ phi   # R^{state_dim}
```

</details>

Swappability means more possibilities than ever before. In Table 1 of our recent preprint, we find that implementing a collection of standard algorithms created a combinatorial space that included novel (i.e., not found in the literature despite search efforts) methods that outperformed existing methods substantially on many of our internal benchmarks (keep an eye out for an upcoming pre-print on this).

## What's up next?

1. Support applied scientists in using dynestyx in their workflows
2. Bring in methodologists to better disseminate their ever-evolving cutting edge work
3. Create a prescriptive and iterative workflow of best practices for how practitioners should go about answering questions around forecasting, system identification, and inference, following in the footsteps of the Bayesian Workflow[^bayesian-workflow]


## Common questions



**Wait, can't uncle AI just code this stuff up for me automagically whenever we need it?**

- Yes, but will you ever really trust it? When you vibe-code, would you rather see a matrix factorization done with LAPACK / scipy or in 50 lines of new never-before-used AI-code? Our vision for interacting with AI-based coding is to continue the work of consolidating and verifying computational tools, and "caching" them---it is wasteful, uninterpretable, and error-prone to re-create such programs from scratch every time we embark on a new project.



**Wait, isn't diversity good and monolith bad?**

- Yes, but Dynestyx is not a monolith! It does adhere to a general underpinning mathematical/statistical framework, but we believe that this offers valuable shared ground that will enable easy comparisons, swappability, and composability across methods and domains. If a problem does not fit within the framework, it is the job of the existing community to wrestle with this challenge and either find a way to expand its framework or simply support (compassionately) from afar.

[^bayesian-workflow]: Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., ... & Modrák, M. (2020). Bayesian workflow. arXiv preprint arXiv:2011.01808.

[^stan]: Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., ... & Riddell, A. (2017). Stan: A probabilistic programming language. Journal of Statistical Software, 76, 1-32.

[^fish-telemetry]: Lavender, E., et al. (2025). [Particle algorithms for animal movement modelling in receiver arrays](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70028). *Methods in Ecology and Evolution*, 16(8), 1808–1819.

[^patter]: Lavender, E., Scheidegger, A., Albert, C., Biber, S. W., Illian, J., Thorburn, J., Smout, S., & Moor, H. (2025). [patter: Particle algorithms for animal tracking in R and Julia](https://doi.org/10.1111/2041-210X.70029). *Methods in Ecology and Evolution*, 16, 1609–1616.

[^dynestyx-preprint]: Waxman, D., Batenkov, D., Feser, J., Zane, A., Bingham, E., Marzouk, Y., & Levine, M. E. (2026). [Dynestyx: A Probabilistic Programming Library for Dynamical Systems](https://arxiv.org/abs/2606.16985). arXiv preprint arXiv:2606.16985.

[^covid-forecasts]: Bracher, J., et al. (2021). [A pre-registered short-term forecasting study of COVID-19 in Germany and Poland during the second wave](https://doi.org/10.1038/s41467-021-25207-0). *Nature Communications*, 12, 5173. See Figures 2–3 (October–December 2020).
