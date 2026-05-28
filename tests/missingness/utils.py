import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist


def set_full_row_missing(obs_values, row_idx, *, member_idx=None):
    if member_idx is None:
        return obs_values.at[row_idx].set(jnp.nan)
    return obs_values.at[member_idx, row_idx].set(jnp.nan)


def set_partial_row_missing(obs_values, row_idx, dim_idx=0, *, member_idx=None):
    if member_idx is None:
        return obs_values.at[row_idx, dim_idx].set(jnp.nan)
    return obs_values.at[member_idx, row_idx, dim_idx].set(jnp.nan)


def manual_masked_mvn_log_prob(mu, cov, y, obs_mask):
    obs_mask_np = np.asarray(obs_mask, dtype=bool)
    if obs_mask_np.ndim == 0:
        if not obs_mask_np.item():
            return jnp.array(0.0)
        return dist.Normal(mu, jnp.sqrt(cov)).log_prob(y)

    if not obs_mask_np.any():
        return jnp.array(0.0, dtype=jnp.asarray(mu).dtype)

    mu_obs = jnp.asarray(mu)[obs_mask_np]
    y_obs = jnp.asarray(y)[obs_mask_np]
    cov_obs = jnp.asarray(cov)[np.ix_(obs_mask_np, obs_mask_np)]
    return dist.MultivariateNormal(mu_obs, covariance_matrix=cov_obs).log_prob(y_obs)


def manual_masked_independent_normal_log_prob(loc, scale, y, obs_mask):
    obs_mask_np = np.asarray(obs_mask, dtype=bool)
    if obs_mask_np.ndim == 0:
        if not obs_mask_np.item():
            return jnp.array(0.0)
        return dist.Normal(loc, scale).log_prob(y)

    if not obs_mask_np.any():
        return jnp.array(0.0, dtype=jnp.asarray(loc).dtype)

    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    y = jnp.asarray(y)
    per_dim = dist.Normal(loc, scale).log_prob(y)
    return jnp.sum(per_dim[obs_mask_np])


def latent_conditioning_data(trace):
    return {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and "_x_" in name
    }


def factor_log_prob(trace, name):
    return jnp.asarray(trace[name]["fn"].log_factor)


def observation_site_names(trace, *, prefix="f_y"):
    return sorted(
        name for name in trace if name.startswith(prefix) and not name.endswith("_lp")
    )


def observation_log_probs(trace, *, prefix="f_y"):
    pieces = []
    scalar_name = f"{prefix}_0_lp"
    if scalar_name in trace:
        pieces.append(jnp.atleast_1d(factor_log_prob(trace, scalar_name)))

    scan_names = sorted(
        name
        for name in trace
        if name.startswith(prefix) and name.endswith("_lp") and name != scalar_name
    )
    pieces.extend(jnp.ravel(factor_log_prob(trace, name)) for name in scan_names)

    if not pieces:
        raise KeyError(f"No observation log-prob sites found for prefix {prefix!r}.")

    return jnp.concatenate(pieces)
