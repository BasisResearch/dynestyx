# Handlers

`dynestyx` is built using [`effectful`](https://github.com/BasisResearch/effectful), which operates using a primitive called a `handler`. The details of this can be abstracted away from the typical user experience, but impacts the implementation of the `sample` primitive. The long story short is that the basic implementation of `sample` is empty, and it is actually "interpreted" by context. For **hierarchical** models with multiple trajectories, use [`plate`](#plate) together with NumPyro sampling inside the plate context. For example,

```python
with Filter(EKFConfig()):
    dsx.sample("f", dynamical_model, obs_times=obs_times, obs_values=obs_values)
```

will implement the `dsx.sample` primitive using an extended Kalman filter. For more details, see the corresponding [developer API page](../developer/handlers.md).

::: dynestyx.handlers
    options:
        members:
            - sample
            - plate
