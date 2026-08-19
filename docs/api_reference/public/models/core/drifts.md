# Drift

`Drift` objects define the deterministic term $f$ in a continuous-time state evolution
$$
\begin{aligned}
dx_t &= f(x_t, u_t, t)\,dt + \sigma(x_t, u_t, t)\,dW_t,
\end{aligned}
$$
where the stochastic term $W_t$ may be absent, yileding an ODE.

## Drift
::: dynestyx.models.drifts.Drift
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Potential
::: dynestyx.models.drifts.Potential
    options:
      show_root_heading: false
      show_root_toc_entry: false

## AffineDrift
::: dynestyx.models.drifts.AffineDrift
    options:
      show_root_heading: false
      show_root_toc_entry: false

## ImExDrift
::: dynestyx.models.drifts.ImExDrift
    options:
      show_root_heading: false
      show_root_toc_entry: false



