# PreparedModel

`PreparedModel` wraps a NumPyro model callable and retains the static observation
missingness layouts needed by `LatentPathBuilder`. This works with automatic LPB
selection and with explicit LPB handlers, even if the model constructs new
`DynamicalModel` objects from sampled parameters on every execution.

Prepare the wrapper with concrete observations before using traced model
arguments:

```python
import jax.random as jr
from numpyro.infer import MCMC, NUTS

import dynestyx as dsx

model = dsx.PreparedModel(model_fn)
model.prepare(obs_times=times, obs_values=observations)

mcmc = MCMC(
    NUTS(model),
    num_warmup=200,
    num_samples=200,
    jit_model_args=True,
)
mcmc.run(jr.PRNGKey(1), obs_times=times, obs_values=observations)
```

`prepare` executes the full model once under a NumPyro seed (zero by default,
configurable through `rng_seed`), including its normal handlers and side effects.
It discards the execution outputs and returns the wrapper. Subsequent model
execution retains the usual seed requirements. An ordinary eager call under a
seed also prepares layouts; JAX does not automatically perform this eager pass.

Each wrapper owns its cache. Entries are distinguished by sample-site name and
observation shape; plate members have separate entries. Observation times and
finite values may change, but the missingness pattern at an existing site and
shape must remain fixed. A changed pattern raises a layout mismatch error. For a
different pattern, create a new wrapper and a new inference or compiled callable.
Calling `prepare` again does not invalidate existing JAX compilations or reset
the wrapper's layouts.

Explicit `missing_obs_metadata` takes precedence for that call and does not
replace the wrapper's cached layout. The wrapper stores no parameter draws,
inferred states, or model outputs. Nested wrapped models use their own caches,
and the caller's cache scope is restored when a model returns or raises.

::: dynestyx.prepared.PreparedModel
    options:
        members:
            - __init__
            - __call__
            - prepare
