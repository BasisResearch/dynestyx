# HMM
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_hmm_states_and_observations(
    times,
    x,
    y,
    state_cmap="tab10",
    obs_cmap="Set1",
    show_fig=False,
    save_path=None,
    obs_style="auto",
    obs_marker="x",
):
    """
    Plot latent discrete HMM states as colored background bands
    with observed signals overlaid.

    :param times: (T,) Time points
    :param x: (T,) Discrete latent state indices (0..K-1)
    :param y: (T,) or (T, N_obs) Observations
    """

    times = np.asarray(times)
    x = np.asarray(x)
    y = np.asarray(y)

    T = len(times)
    if x.shape[0] != T:
        raise ValueError(f"`x` must have shape (T,), got {x.shape} with T={T}.")
    if y.shape[0] != T:
        raise ValueError(
            f"`y` must have shape (T,) or (T, N_obs), got {y.shape} with T={T}."
        )

    # ---- Normalize observation shape ----
    if y.ndim == 1:
        y = y[:, None]  # (T, 1)

    N_obs = y.shape[1]

    # ---- Discrete state labels (may not be 0..K-1) ----
    state_values = np.unique(x)
    K = int(state_values.size)
    state_to_idx = {int(s): i for i, s in enumerate(state_values.tolist())}

    # ---- Time "edges" for clean contiguous state bands ----
    # For irregular sampling, use midpoints between times; extend at ends by half-step.
    if T == 1:
        dt = 1.0
        edges = np.array([times[0] - 0.5 * dt, times[0] + 0.5 * dt])
    else:
        mids = 0.5 * (times[:-1] + times[1:])
        left = times[0] - 0.5 * (times[1] - times[0])
        right = times[-1] + 0.5 * (times[-1] - times[-2])
        edges = np.concatenate(([left], mids, [right]))

    # ---- Color maps ----
    cmap_states = plt.cm.get_cmap(state_cmap, K)
    state_colors = [cmap_states(k) for k in range(K)]

    cmap_obs = plt.cm.get_cmap(obs_cmap, N_obs)
    obs_colors = [cmap_obs(i) for i in range(N_obs)]

    fig, ax = plt.subplots(figsize=(10, 4))

    # ---- Draw state background as contiguous segments ----
    def draw_state_blocks():
        start = 0
        for t in range(1, T + 1):
            if t == T or x[t] != x[start]:
                s_val = int(x[start])
                k = state_to_idx[s_val]
                ax.axvspan(
                    edges[start],
                    edges[t],
                    color=state_colors[k],
                    alpha=0.18,
                    linewidth=0,
                )
                start = t

    draw_state_blocks()

    # ---- Choose observation style ----
    # If observations are discrete-valued, lines look misleading; default to scatter.
    def _is_discrete_column(col: np.ndarray) -> bool:
        if np.issubdtype(col.dtype, np.integer) or np.issubdtype(col.dtype, np.bool_):
            return True
        # Heuristic: "few unique values" relative to length suggests discrete categories.
        # (Keeps continuous floats like SDE outputs as lines.)
        unique = np.unique(col)
        return unique.size <= min(20, max(3, T // 5))

    if obs_style not in {"auto", "line", "scatter"}:
        raise ValueError("`obs_style` must be one of {'auto','line','scatter'}.")

    # ---- Plot observations ----
    for n in range(N_obs):
        col = y[:, n]
        style = obs_style
        if style == "auto":
            style = "scatter" if _is_discrete_column(col) else "line"

        if style == "line":
            ax.plot(
                times,
                col,
                color=obs_colors[n],
                lw=2,
                label=f"obs[{n}]",
                zorder=5,
            )
        else:
            ax.scatter(
                times,
                col,
                color=obs_colors[n],
                marker=obs_marker,
                s=35,
                linewidths=1.5,
                label=f"obs[{n}]",
                zorder=6,
            )

    # ---- Formatting ----
    ax.set_xlabel("Time")
    ax.set_ylabel("Observations")
    ax.set_title("HMM latent states and observations")

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    # ---- Build state legend separately ----
    from matplotlib.patches import Patch

    state_patches = [
        Patch(
            facecolor=state_colors[state_to_idx[int(s)]],
            alpha=0.3,
            label=f"state {int(s)}",
        )
        for s in state_values
    ]

    ax.legend(
        handles=state_patches + ax.get_legend_handles_labels()[0],
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif show_fig:
        plt.show()

    return fig, ax


def plot_continuous_states_and_partial_observations(
    times, x, y, show_fig=False, save_path=None
):
    """
    Plot continuous latent states with partial noisy observations.

    :param times: (T,) Time points
    :param x: (T, state_dim) Continuous latent states
    :param y: (T, obs_dim) Observations
    :param show_fig: Whether to show the figure
    :param save_path: Optional path to save the figure
    """
    times = np.asarray(times)
    x = np.asarray(jnp.asarray(x))
    y = np.asarray(jnp.asarray(y))

    T, num_x = x.shape
    num_y = y.shape[1]

    # Colors
    state_color = "C0"
    obs_color = "C2"

    # Figure
    fig, axes = plt.subplots(
        num_x, 1, figsize=(10, 2.2 * num_x), sharex=True, constrained_layout=True
    )

    if num_x == 1:
        axes = [axes]

    # Plot
    for i, ax in enumerate(axes):
        # Latent state
        is_first_state = i == 0
        ax.plot(
            times,
            x[:, i],
            color=state_color,
            lw=2.0,
            alpha=0.95,
            label="Latent state" if is_first_state else None,
        )

        # Observations (assume first num_y states are observed)
        if i < num_y:
            is_first_obs = i == 0
            ax.scatter(
                times,
                y[:, i],
                s=28,
                facecolors="none",
                edgecolors=obs_color,
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
                label="Observation" if is_first_obs else None,
            )

        ax.set_ylabel(f"x{i + 1}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")

    # Legend
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif show_fig:
        plt.show()

    return fig, axes
