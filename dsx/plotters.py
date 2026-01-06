import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
