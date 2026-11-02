import matplotlib.pyplot as plt
import numpy as np
from veery.accessories import percent_diff, robustness
from pathlib import Path


def plot_rollouts(histories, results, save_dir=None, ylabel="Actual Value"):
    """Plots actual (true) value vs time for a set of rollouts, colored green (pass) or red (fail).

    Args:
        histories (list[pd.DataFrame]): rollout histories, each with 'Time' and 'Actual'
        results (np.ndarray[bool]): pass/fail per history, same order and length as histories
        save_dir (str, opt): path to the folder in which to save the file. Defaults to None and doesn't save figure.
        ylabel (str, opt): y-axis label text. Defaults to "Actual Value" without units.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for history, passed in zip(histories, results):
        color = "green" if passed else "red"
        ax.plot(
            history["Time"],
            history["Actual"],
            color=color,
            alpha=0.5,
            linewidth=0.8,
        )

    ax.set_xlabel("Time, [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Failures: {len(results) - results.sum()}/{len(results)}")
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
        plt.savefig(f"{save_dir}/rollouts.png", dpi=300)

    return fig


def plot_overlapping_rollouts(
    histories_list,
    results_list,
    sigmas,
    colors,
    threshold=0.03,
    save_dir=None,
    ylabel="Value of Interest",
):
    """Plots power vs time for a set of rollouts, colored green (pass) or red (fail).

    Args:
        histories_list (list[pd.DataFrame]): list of dataframes containing rollout histories, each with 'Time' ,'Desired' and 'Actual'
        results_list (np.ndarray[bool]): pass/fail per history, same order and length as histories
        sigmas (list): standard deviations for the disturbance distribution
        colors (list): list of matplotlib colors the same length as sigmas
        threshold (float): percent error tolerance for any given timestep
        save_dir (str, opt): file path for location to save the file. Defaults to None and doesn't save the file.
        ylabel (str): the name of the output variable used in the spec calculation. Note: this is directly used for the y-axis label, format accordingly.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    desired = histories_list[0][0]["Desired"]
    time = histories_list[0][0]["Time"]
    # ax.plot(history["time"], desired, "k--", label="Prescribed Power Profile")
    if threshold is not None:
        ax.fill_between(
            time,
            desired * (1 - threshold),
            desired * (1 + threshold),
            color="green",
            alpha=0.15,
            label="Specification Region",
        )

    for histories, results, sig, color in zip(
        histories_list, results_list, sigmas, colors
    ):
        failures = len(results) - results.sum()

        for i, history in enumerate(histories):
            color = color
            if i == 0:
                ax.plot(
                    history["Time"],
                    history["Actual"],
                    color=color,
                    alpha=0.5,
                    linewidth=0.8,
                    label=r"$\sigma=$" + f"{sig}, Failures={failures}/{len(results)}",
                )
            else:
                ax.plot(
                    history["Time"],
                    history["Actual"],
                    color=color,
                    alpha=0.5,
                    linewidth=0.8,
                )

    ax.set_xlabel("Time, [s]")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
        plt.savefig(f"{save_dir}/overlapping_rollouts.png", dpi=400)

    return fig


def plot_tree(tree, profile=None, threshold=None, save_path=None):
    """_summary_

    Args:
        tree (_type_): _description_
        profile (_type_, optional): _description_. Defaults to None.
        threshold (_type_, optional): _description_. Defaults to None.
        save_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    max_n = max(n.N for n in tree)
    for node in tree:
        if node.parent is None:
            continue
        w = node.N / max_n  # visit share, 0-1
        ax.plot(
            [node.parent.state["time"], node.state["time"]],
            [node.parent.state["_p"], node.state["_p"]],
            color="red" if node.path_rho < 0 else "tab:blue",
            alpha=0.3 + 0.7 * w,
            linewidth=0.5 + 2.5 * w,
            marker=".",
        )
    if profile is not None:
        t_max = max(n.state["time"] for n in tree)
        ts = np.linspace(0, t_max, 200)
        desired = np.array([float(profile(t)) for t in ts])
        ax.plot(ts, desired, "k--", label="Prescribed Power Profile")
        if threshold is not None:
            ax.fill_between(
                ts,
                desired * (1 - threshold),
                desired * (1 + threshold),
                color="green",
                alpha=0.15,
                label="Specification Region",
            )
        ax.legend()
    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Power Fraction, [-]")
    # ax.set_title(f"MCTS tree ({len(tree)} nodes)")
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
    return fig, ax


def plot_tree_structure(tree, save_path=None):
    # y layout: leaves get consecutive rows, parents sit at the mean of their children
    # (iterative post-order, since episode depth can exceed Python's recursion limit)
    y, slot = {}, 0
    stack = [(tree[0], False)]
    while stack:
        node, visited = stack.pop()
        if node.children and not visited:
            stack.append((node, True))
            stack.extend((c, False) for c in reversed(node.children))
        elif node.children:
            y[node] = np.mean([y[c] for c in node.children])
        else:
            y[node] = slot
            slot += 1

    fig, ax = plt.subplots(figsize=(10, 16))
    for n in tree:
        if n.parent is not None:
            ax.plot(
                [n.parent.state["time"], n.state["time"]],
                [y[n.parent], y[n]],
                color="lightgray",
                linewidth=0.8,
                zorder=1,
            )

    max_n = max(n.N for n in tree)
    sc = ax.scatter(
        [n.state["time"] for n in tree],
        [y[n] for n in tree],
        s=[15 + 200 * n.N / max_n for n in tree],
        c=[n.Q for n in tree],
        cmap="RdYlGn",
        zorder=2,
    )
    failed = [n for n in tree if n.parent is not None and n.path_rho < 0]
    if failed:
        ax.scatter(
            [n.state["time"] for n in failed],
            [y[n] for n in failed],
            marker="x",
            color="red",
            s=40,
            zorder=3,
            label="Failure",
        )
        ax.legend()

    fig.colorbar(sc, label="Q")
    ax.set_xlabel("Time, [s]")
    ax.set_yticks([])
    # ax.set_title(f"MCTS structure ({len(tree)} nodes)")
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
    return fig, ax


def plot_robustness(history, threshold=0.03, time=None, ax=None, save_dir=None):
    """Plot absolute percent power deviation for one rollout with spec bound and rho arrow."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    t = history.index if time is None else time

    rho_gen = robustness(
        p_actual=history["Actual"],
        p_desired=history["Desired"],
        threshold=threshold,
    )

    pct_dev = percent_diff(history["Actual"], history["Desired"])

    # robustness: smallest margin to the spec (unclipped, so violations show as negative)
    margins = threshold - pct_dev
    i = np.argmin(margins)
    rho = margins[i]

    print(f"rho from robustness calc: {rho_gen}, rho from built-in: {rho}")

    ax.plot(t, pct_dev)
    ax.axhline(threshold, color="r", linestyle="-")
    ymax = max(threshold * 1.5, pct_dev.max() * 1.1)
    ax.axhspan(threshold, ymax, color="r", alpha=0.2)
    ax.set_ylim(top=ymax)

    # arrow from spec line to the worst-case point
    ax.annotate(
        "",
        xy=(t[i], pct_dev[i]),
        xytext=(t[i], threshold),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.5, shrinkA=0, shrinkB=0),
    )
    ax.text(t[i], (threshold + pct_dev[i]) / 2, r"$\rho$ ", ha="right", va="center")

    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Fractional Power Deviation, [-]")
    if save_dir:
        plt.savefig(f"{save_dir}/robustness.png", dpi=400, bbox_inches="tight")

    return ax
