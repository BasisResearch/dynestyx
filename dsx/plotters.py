import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# HMM
def plot_hmm_states_and_observations(times, x, y, mu, sigma):
    K = len(mu)
    colors = ["#AED6F1", "#ABEBC6", "#F9E79F"]  # soft, readable

    fig, ax = plt.subplots(figsize=(10, 3))

    # --- State background shading ---
    for k in range(K):
        mask = (x == k)
        ax.fill_between(
            times,
            y.min() - 3 * sigma,
            y.max() + 3 * sigma,
            where=mask,
            color=colors[k],
            alpha=0.4,
            step="post",
        )

    # --- Observations ---
    ax.plot(times, y, color="black", lw=1, label="observations")

    # --- Emission means ---
    for k in range(K):
        ax.axhline(mu[k], color=colors[k], lw=2, linestyle="--")

    # Legend
    patches = [
        mpatches.Patch(color=colors[k], label=f"State {k}")
        for k in range(K)
    ]
    ax.legend(handles=patches, loc="upper left", frameon=False)

    ax.set_xlabel("Time")
    ax.set_ylabel("Observation y")
    ax.set_title("Latent state regimes with observations")
    plt.tight_layout()
    plt.show()


def plot_continuous_states_and_partial_observations(
    times,
    x,
    y,
    x_labels=None,
    y_indices=None,
    show_fig=False,
    figsize=(10, 2.2),
):
    """
    Clean, publication-quality plot of continuous latent states
    with partial noisy observations (Seaborn-styled).
    """

    # ---------- Style ----------
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.4,
        },
    )

    # ---------- Data prep ----------
    times = np.asarray(times)
    x = np.asarray(jnp.asarray(x))
    y = np.asarray(jnp.asarray(y))

    T, num_x = x.shape
    num_y = y.shape[1]

    if y_indices is None:
        print("Assuming observations correspond to first num_y states.")
        y_indices = list(range(num_y))

    if x_labels is None:
        x_labels = [f"x{i+1}" for i in range(num_x)]

    # Colors
    state_color = sns.color_palette("deep")[0]
    obs_color = sns.color_palette("deep")[2]

    # ---------- Figure ----------
    fig, axes = plt.subplots(
        num_x, 1,
        figsize=(figsize[0], figsize[1] * num_x),
        sharex=True,
        constrained_layout=True,
    )

    if num_x == 1:
        axes = [axes]

    # ---------- Plot ----------
    for i, ax in enumerate(axes):
        # Latent state
        is_first_state = (i == 0)
        ax.plot(
            times,
            x[:, i],
            color=state_color,
            lw=2.0,
            alpha=0.95,
            label="Latent state" if is_first_state else None,
        )

        # Observations
        if i in y_indices:
            is_first_obs = (i == y_indices[0])
            y_idx = y_indices.index(i)
            ax.scatter(
                times,
                y[:, y_idx],
                s=28,
                facecolors="none",          # hollow
                edgecolors=obs_color,
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
                label="Observation" if is_first_obs else None,
            )

        ax.set_ylabel(x_labels[i])

    axes[-1].set_xlabel("Time")

    # ---------- One clean legend ----------
    axes[0].legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=2,
    )

    if show_fig:
        plt.show()

    return fig, axes
