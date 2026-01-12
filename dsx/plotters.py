# HMM
import matplotlib.pyplot as plt
import numpy as np


def plot_hmm_states_and_observations(
    times, x, y, state_cmap="tab10", obs_cmap="Set1", show_fig=False, save_path=None
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

    # ---- Normalize observation shape ----
    if y.ndim == 1:
        y = y[:, None]  # (T, 1)

    N_obs = y.shape[1]
    K = int(x.max()) + 1

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
                k = int(x[start])
                ax.axvspan(
                    times[start],
                    times[t - 1] if t < T else times[-1],
                    color=state_colors[k],
                    alpha=0.18,
                    linewidth=0,
                )
                start = t

    draw_state_blocks()

    # ---- Plot observations ----
    for n in range(N_obs):
        ax.plot(
            times,
            y[:, n],
            color=obs_colors[n],
            lw=2,
            label=f"obs[{n}]",
            zorder=5,
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
        Patch(facecolor=state_colors[k], alpha=0.3, label=f"state {k}")
        for k in range(K)
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
