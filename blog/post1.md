Here at Basis, one of our favorite things is to identify and address unmet computational needs that permeate across sectors of science and engineering. It allows us to play a uniquely catalyzing role in research that most institutions are unable to incentivize. Roughly one year ago, we set out on one such mission, with a laser focus on improving how applied and methodological researchers/scientists/engineers work with data and models from time-evolving (dynamical) systems to answer key questions like:

1. What will happen next in this system? (sometimes called forecasting)
2. Where has the system been? (sometimes generically called inference, or smoothing in certain sub-communities that are dear to my heart)
3. How does the system work / what are its rules? (sometimes called system identification)
4. What will happen if I intervene on the system (sometimes called counterfactual inference)?

Good answers to these questions are paired with quantified uncertainty---that is, how sure are we about the answer? The best answers also address the meta-question: how reliable is the assessment of uncertainty? For those of us following forecasts of early COVID outbreaks, you may remember that forecasts from each team/organization came with "error bars"; and those error bars often did not even overlap when looking across teams---clearly something was wrong!

Countless fields rely on high-quality answers to these questions, from epidemiology to robotics engineering, and from atmospheric sciences to biomedical device engineering. Supporting these applied needs is a large and robust community of methodological researchers in fields that include applied mathematics, statistics, computer science, and engineering.

However, the interactions within and between these applied and methodological fields alike leaves much to be desired; in particular, a suite of reliable, efficient, and general purpose tools that can serve the majority of evolving needs across these fields.

## Our assessment.

There is a critical lack of reliable, efficient, and general purpose tools that can serve the majority of evolving needs across these fields. It is common for a methodological research group to maintain a python package that deploys their latest or most used algorithms. However, there are hundreds of these that live in isolation (each with their own scope), creating frustrating incompatibilities in software, mathematical underpinnings, and data particularities. In applied domains, it is common for a field to quickly converge on a specific tool with a (sometimes modest) success history. It is also common for individual groups to re-implement generic toolings in a field-specific lens.

The result is dozens of brittle echo-chambers in which comparing methods within a given problem is unnecessarily difficult and where methodologists are needlessly disconnected from challenges specific to particular applied domains.

A great example I came across recently is this paper[^fish-telemetry] which uses `patter`[^patter], a Julia-based package for Particle Filtering that is specifically focused on ecological monitoring via acoustic telemetry data; it makes some specific modeling choices that are highly relevant to the setting, but the algorithm itself (while written nicely) is quite standard (and, in fact, does not take advantage of recent advances in SOTA PFs). With the right tooling in place, one might view this repository similarly to one with a custom implementation of linear regression specially designed for measurements of the microbiome. To be clear, I think this work is great---I really like that paper! But what happens as state-of-the-art filtering improves? Is it really the responsibility of ecological scientists to stay up to date on the latest and greatest in Particle Filtering? I hope not! If only there were a general tool maintained and updated by a collective community of methodological and applied researchers that ensures cutting-edge, generalist support for the most commonly used algorithms and practices in dynamical systems modeling and inference?!

## This is why we built Dynestyx[^dynestyx-preprint].

We introduce `dynestyx`, a probabilistic programming library that treats dynamical systems as first-class objects. `dynestyx` builds on top of `numpyro` to provide a clean, unified interface for Bayesian state-space models, providing a one-stop-shop for parameter inference and state inference.

[GitHub](https://github.com/BasisResearch/dynestyx) · [Docs](https://basisresearch.github.io/dynestyx/stable/) · [Preprint](https://arxiv.org/abs/2606.16985)

Our goal in building `dynestyx` is two-fold, serving both practitioners and theoreticians. On the side of application, `dynestyx` provides an approachable interface for model-building, remaining expressive over model choices and providing a variety of state-of-the-art inference methods. On the side of methodology, `dynestyx` offers a natural integration surface to implement, test, and apply new algorithmic approaches in real-world problems. This is reminiscent of `stan`[^2], which has spurred both applied work in Bayesian statistics and methodological work in Monte Carlo and variational inference.

## A unified interface

Core to `dynestyx` is its unified interface for a state-space model (SSM, defined below). In particular, `dynestyx` supports inference across a wide variety of model classes belonging to the SSM family. A core thesis of ours is that dynestyx should look a lot like the math you use to describe your model, so let's start there.

### Mathematical description of an SSM

A state-space model concerns the evolution of a *latent state* $x_t \in\mathbb{R}^{d_x}$, beginning with a possibly uncertain initial condition:
$$x_0 \sim \pi_0$$
The subsequent evolution may be deterministic or stochastic, occur continuously or discretely in time, and may have explicit dependent on time $t \in \mathbb{R}^+$ or a sequence of external control inputs $u_t \in \mathbb{R}^{d_u}$. Let's consider the stochastic case for now, since it's more general. In discrete-time, our state then evolves according to a Markov chain with some *rules* $\theta$ :

$$
x_t \sim p(x_t | x_{t-1}, u_{t-1}, t-1, t; \ \theta).
$$

In continuous time, we instead have a stochastic differential equation (SDE):

$$
\mathrm{d}x_t = f(x_t, u_t, t)  \mathrm{d}t + g(x_t, u_t, t)  \mathrm{d}W_t,
$$

where $W_t$ is a $d_b$-dimensional Brownian motion, \[f : \mathbb{R}^{d_x} \times \mathbb{R}^{d_u} \times \mathbb{R}_{+} \to \mathbb{R}^{d_x}\] is the drift function (governing deterministic dynamics), and $g : \mathbb{R}^{d_x} \times \mathbb{R}^{d_u} \times \mathbb{R}_{+} \to \mathbb{R}^{d_x \times d_b}$ is the diffusion function (governing the coupling to the stochastic part of the evolution).

### Mathematics to Code

Given the mathematical description of a state-space model, it is straightforward to translate to a `dynestyx` model! The key abstraction in `dynestyx` is a `DynamicalModel`, which takes as input exactly the data we specified above: an initial condition (`initial_condition`), a state evolution (`state_evolution`), and an observation model (`observation_model`):

|                 | stochastic                                                                                                                | deterministic                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| continuous-time | $\mathrm{d}x_t = f(x_t, u_t, t) \, \mathrm{d}t + g(x_t, u_t, t) \, \mathrm{d}W_t$<br><br>`ContinuousTimeStateEvolution(f, g)` | $\mathrm{d}x_t = f(x_t, u_t, t) \, \mathrm{d}t$<br><br>`ContinuousTimeStateEvolution(f)`               |
| discrete-time   | $x_t \sim p(x_t \mid x_{t-1}, u_{t-1}, t-1, t;\,\theta)$<br><br>`DiscreteTimeStateEvolution(p)`                           | $x_t = f(x_{t-1}, u_{t-1}, t-1, t;\,\theta)$<br><br>`DiscreteTimeStateEvolution(DiracTransition(f))` |

## Let's see Dynestyx in action!

[Gallery]

## Common questions



**Wait, can't uncle AI just code this stuff up for me automagically whenever we need it?**

- Yes, but will you ever really trust it? When you vibe-code, would you rather see a matrix factorization done with LAPACK / scipy or in 50 lines of new never-before-used AI-code? Our vision for interacting with AI-based coding is to continue the work of consolidating and verifying computational tools, and "cacheing" them---it is wasteful, uninterpretable, and error-prone to re-create such programs from scratch every time we embark on a new project.



**Wait, isn't diversity good and monolith bad?**

- Yes, but Dynestyx is not a monolith! It does adhere to a general underpinning mathematical/statistical framework, but we believe that this offers valuable shared ground that will enable easy comparisons, swappability, and composability across methods and domains. If a problem does not fit within the framework, it is the job of the existing community to wrestle with this challenge and either find a way to expand its framework or simply support (compassionately) from afar. Dynestyx

[^1]: Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., ... & Modrák, M. (2020). Bayesian workflow. arXiv preprint arXiv:2011.01808.

[^2]: Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., ... & Riddell, A. (2017). Stan: A probabilistic programming language. Journal of Statistical Software, 76, 1-32.

[^fish-telemetry]: Futia, M. H., Binder, T. R., Henderson, M. J., & Marsden, J. E. (2024). [Modeling regional occupancy of fishes using acoustic telemetry: A model comparison framework applied to lake trout](https://www.usgs.gov/publications/modeling-regional-occupancy-fishes-using-acoustic-telemetry-a-model-comparison). *Animal Biotelemetry*. [https://doi.org/10.1186/s40317-024-00380-3](https://doi.org/10.1186/s40317-024-00380-3).

[^patter]: Lavender, E., Scheidegger, A., Albert, C., Biber, S. W., Illian, J., Thorburn, J., Smout, S., & Moor, H. (2025). [patter: Particle algorithms for animal tracking in R and Julia](https://doi.org/10.1111/2041-210X.70029). *Methods in Ecology and Evolution*, 16, 1609–1616.

[^dynestyx-preprint]: Waxman, D., Batenkov, D., Feser, J., Zane, A., Bingham, E., Marzouk, Y., & Levine, M. E. (2026). [Dynestyx: A Probabilistic Programming Library for Dynamical Systems](https://arxiv.org/abs/2606.16985). arXiv preprint arXiv:2606.16985.
