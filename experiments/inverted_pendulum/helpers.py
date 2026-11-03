import matplotlib.pyplot as plt
import numpy as np


def clamp(number, min, max):
    if number < min:
        return min
    elif number > max:
        return max
    else:
        return number


def plot_observation_disturbances(xs, sigma):
    fig, ax = plt.subplots()
    for x in xs:
        ax.scatter(x.xo[0, 0], x.xo[1, 0], color="orange")
    ax.set_xlabel(r"xo, $\theta$")
    ax.set_ylabel(r"xo, $\omega$")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.savefig(f"../../results/disturbance_plot_{sigma}.svg", dpi=300)
    return ax


def plot_trajectories(theta_matrix, sigma):
    fig, ax = plt.subplots()
    ax.plot(theta_matrix, color="steelblue", alpha=0.5, linewidth=0.8)
    plt.axhline(np.pi / 4, color="red", linewidth=1)
    plt.axhline(-np.pi / 4, color="red", linewidth=1)
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.xlabel("t, [s]")
    plt.ylabel(r"$\theta$, [rad]")
    plt.savefig(f"../../results/inverted_pendulum_{sigma}.svg", dpi=300)
    return ax
