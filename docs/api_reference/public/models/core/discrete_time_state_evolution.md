# DiscreteTimeStateEvolution

::: dynestyx.models.core.DiscreteTimeStateEvolution
    options:
      show_root_heading: false
      show_root_toc_entry: false

## All-pairs transition scores

Use `vmap` to score every pair of previous and current states:

```python
import jax


def score_from(previous_state):
    transition = dynamics.state_evolution(
        x=previous_state, u=control, t_now=t_now, t_next=t_next
    )
    return jax.vmap(transition.log_prob, out_axes=-1)(states)


pairwise_log_prob = jax.vmap(score_from, out_axes=-2)(previous_states)
```

The result has shape `(*log_prob_batch, n_previous, n_current)`, preserving
distribution batch axes. Dense evaluation requires work and output storage
proportional to `n_previous * n_current` per batch member.

For joint state-path and observation scoring, pass the complete discrete path
to [log_prob](../../handlers.md#pure-apis) as `state_path_params`.
