# Discretizers

Using continuous state evolutions can be inconvenient as they entail running full SDE solves for all transitions.  `dynestyx` provides a set of `Discretizer` objects to discretize a continuous time state evolution to a discrete time state evolution.

::: dynestyx.discretizers
    options:
      members:
        - Discretizer
        - euler_maruyama
        - frozen_jacobian_gaussian
        - taylor_moment_gaussian
        - simulated_likelihood
