import re

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
    data = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and "_x_" in name
    }

    grouped = {}
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        match = re.fullmatch(r"(.+)_x_(\d+)", name)
        if match is None:
            continue
        prefix, idx = match.groups()
        grouped.setdefault(prefix, {})[int(idx)] = site["value"]

    for prefix, members in grouped.items():
        latent_idxs = sorted(idx for idx in members if idx > 0)
        if not latent_idxs:
            continue
        data[f"{prefix}_x_latents"] = jnp.stack(
            [members[idx] for idx in latent_idxs],
            axis=1,
        )

    for name, site in trace.items():
        if site["type"] != "sample" or "_x_" not in name or name.endswith("_x_0"):
            continue
        if re.fullmatch(r"(.+)_x_(\d+)", name) is not None:
            continue
        prefix, _, _ = name.partition("_x_")
        value = jnp.asarray(site["value"])
        if value.ndim < 2:
            continue
        data[f"{prefix}_x_latents"] = jnp.swapaxes(value, 0, 1)

    state_sites = {
        name[: -len("_states")]: jnp.asarray(site["value"])
        for name, site in trace.items()
        if site["type"] == "deterministic" and name.endswith("_states")
    }

    for name in list(data):
        if not name.endswith("_x_0"):
            continue
        prefix = name[: -len("_x_0")]
        if f"{prefix}_x_latents" in data:
            continue
        if prefix in state_sites:
            member_states = state_sites[prefix]
        else:
            match = re.fullmatch(r"(.+)_p(\d+(?:_\d+)*)", prefix)
            if match is None:
                continue
            base_prefix, raw_indices = match.groups()
            if base_prefix not in state_sites:
                continue
            plate_idx = tuple(int(part) for part in raw_indices.split("_"))
            member_states = state_sites[base_prefix][plate_idx]
        if member_states.shape[1] <= 1:
            continue
        data[f"{prefix}_x_latents"] = member_states[:, 1:]

    return data


def factor_log_prob(trace, name):
    if trace[name]["type"] == "deterministic":
        return jnp.asarray(trace[name]["value"])
    return jnp.asarray(trace[name]["fn"].log_factor)


def observation_site_names(trace, *, prefix="f_y"):
    return [
        name for name in trace if name.startswith(prefix) and not name.endswith("_lp")
    ]


def observation_log_probs(trace, *, prefix="f_y"):
    pieces = [
        jnp.ravel(factor_log_prob(trace, name))
        for name in trace
        if name.startswith(prefix) and name.endswith("_lp")
    ]

    if not pieces:
        raise KeyError(f"No observation log-prob sites found for prefix {prefix!r}.")

    return jnp.concatenate(pieces)
