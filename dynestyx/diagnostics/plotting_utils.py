# HMM
import jax
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


def plot_drift_field(
    f_true,
    f_learned,
    f_learned_sd=None,
    x1_range=(-3.0, 3.0),
    x2_range=(-3.2, 3.2),
    num_points=50,
    return_rmse=False,
    relative_error=False,
    trajectory=None,
    trajectory_axes="error",
    trajectory_color="red",
    trajectory_lw=1.5,
    trajectory_alpha=0.85,
):
    """
    Plot true vs learned drift fields in 2D state space.

    Args:
        f_true: Callable f(x) -> (2,) array for true drift.
        f_learned: Callable f(x) -> (2,) array for learned drift.
        f_learned_sd: Optional callable f(x) -> (2,) array of learned stddev.
        x1_range: (min, max) range for x1.
        x2_range: (min, max) range for x2.
        num_points: Number of grid points per axis.
        return_rmse: If True, return (fig, rmse).
        relative_error: If True, plot relative error instead of absolute error.
        trajectory: Optional array of shape (T, 2) to overlay.
        trajectory_axes: "error" or "all".
    """
    x1 = jnp.linspace(x1_range[0], x1_range[1], num_points)
    x2 = jnp.linspace(x2_range[0], x2_range[1], num_points)
    X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
    grid_points = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)

    f_true_vals = jax.vmap(f_true)(grid_points)
    f_learned_vals = jax.vmap(f_learned)(grid_points)

    if f_learned_sd is not None:
        f_learned_sd_vals = jax.vmap(f_learned_sd)(grid_points)
        if f_learned_sd_vals.ndim == 3 and f_learned_sd_vals.shape[1] == 1:
            f_learned_sd_vals = f_learned_sd_vals.squeeze(1)
        f1_sd = np.asarray(f_learned_sd_vals[:, 0].reshape(num_points, num_points))
        f2_sd = np.asarray(f_learned_sd_vals[:, 1].reshape(num_points, num_points))
    else:
        f1_sd = f2_sd = None

    f1_true = np.asarray(f_true_vals[:, 0].reshape(num_points, num_points))
    f2_true = np.asarray(f_true_vals[:, 1].reshape(num_points, num_points))
    f1_learned = np.asarray(f_learned_vals[:, 0].reshape(num_points, num_points))
    f2_learned = np.asarray(f_learned_vals[:, 1].reshape(num_points, num_points))

    f1_err = np.abs(f1_learned - f1_true)
    f2_err = np.abs(f2_learned - f2_true)
    if relative_error:
        f1_err /= np.abs(f1_true) + 1e-6
        f2_err /= np.abs(f2_true) + 1e-6

    vlim1 = float(np.max(np.abs(np.concatenate([f1_true.ravel(), f1_learned.ravel()]))))
    vlim2 = float(np.max(np.abs(np.concatenate([f2_true.ravel(), f2_learned.ravel()]))))

    ncols = 4 if f_learned_sd is not None else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        f1_true.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim1,
        vmax=vlim1,
        aspect="auto",
    )
    axes[0, 0].set_title("f1 true")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        f1_learned.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim1,
        vmax=vlim1,
        aspect="auto",
    )
    axes[0, 1].set_title("f1 learned")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(
        f1_err.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="viridis",
        aspect="auto",
    )
    axes[0, 2].set_title("f1 error")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    if f1_sd is not None:
        im3 = axes[0, 3].imshow(
            f1_sd.T,
            origin="lower",
            extent=(*x1_range, *x2_range),
            cmap="magma",
            aspect="auto",
        )
        axes[0, 3].set_title("f1 stddev")
        fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im4 = axes[1, 0].imshow(
        f2_true.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim2,
        vmax=vlim2,
        aspect="auto",
    )
    axes[1, 0].set_title("f2 true")
    fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(
        f2_learned.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim2,
        vmax=vlim2,
        aspect="auto",
    )
    axes[1, 1].set_title("f2 learned")
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(
        f2_err.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="viridis",
        aspect="auto",
    )
    axes[1, 2].set_title("f2 error")
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    if f2_sd is not None:
        im7 = axes[1, 3].imshow(
            f2_sd.T,
            origin="lower",
            extent=(*x1_range, *x2_range),
            cmap="magma",
            aspect="auto",
        )
        axes[1, 3].set_title("f2 stddev")
        fig.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(False)

    if trajectory is not None:
        traj = np.asarray(trajectory)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError("trajectory must have shape (T, 2) for (x1, x2)")
        if trajectory_axes == "error":
            overlay_axes = [axes[0, 2], axes[1, 2]]
        elif trajectory_axes == "all":
            overlay_axes = list(axes.ravel())
        else:
            raise ValueError('trajectory_axes must be "error" or "all"')
        for ax in overlay_axes:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=trajectory_color,
                lw=trajectory_lw,
                alpha=trajectory_alpha,
                zorder=5,
            )

    if return_rmse:
        rmse = float(jnp.sqrt(jnp.mean((f_learned_vals - f_true_vals) ** 2)))
        return fig, rmse
    return fig
