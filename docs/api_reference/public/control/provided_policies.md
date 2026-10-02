# Provided Policies

Any callable matching [`PolicyCallable`](policy_callable.md) can be used as a
control policy. dynestyx provides the following baseline implementations:

| Policy | Description |
|---|---|
| [`MPPI`](mppi.md) | Model Predictive Path Integral control. At each step it samples candidate control sequences around a nominal sequence, rolls each one out through the model, scores them with a user-supplied loss, computes their weighted average and applies. The proposal, the weighting and the state update can be customized by subclassing. |
