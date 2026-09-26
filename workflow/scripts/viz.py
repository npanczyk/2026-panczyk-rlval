import matplotlib.pyplot as plt


def plot_power(history, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["time"], history["desired_power"], label="Desired Power")
    ax.plot(history["time"], history["observed_power"], label="Observed Power")
    ax.plot(history["time"], history["actual_power"], label="Actual Power")
    ax.legend()
    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Fraction of Total Power")
    ax.grid()
    fig.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/power.png", dpi=300)

    return fig

def plot_rollouts(histories, results):
    """Plots power vs time for a set of rollouts, colored green (pass) or red (fail).

    Args:
        histories (list[pd.DataFrame]): rollout histories, each with 'time' and 'actual_power'
        results (np.ndarray[bool]): pass/fail per history, same order and length as histories

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for history, passed in zip(histories, results):
        color = "green" if passed else "red"
        ax.plot(history["time"], history["actual_power"], color=color, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Fraction of Total Power")
    ax.set_title(f"Failures: {len(results) - results.sum()}/{len(results)}")
    fig.tight_layout()

    return fig