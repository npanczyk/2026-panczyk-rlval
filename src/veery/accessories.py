import numpy as np


def check_spec(history, threshold, percent=True):
    """Checks if a rollout's power trajectory stays within 3% (relative) of desired power.

    Args:
        history (pd.DataFrame): rollout history with 'Actual' and 'Desired' columns
        threshold (float): specficiation value for percent difference limit
        percent (bool): True to use a percent deviation from the "Desired" column in history. Otherwise uses threshold as a flat value
    Returns:
        bool: True if actual power never deviates >=3% (relative) from desired power
    """
    if percent:
        dev = np.abs((history["Actual"] - history["Desired"]) / history["Desired"])
    else:
        dev = np.abs((history["Actual"]))
    return (dev < threshold).all()


def percent_diff(actual, desired):
    return abs(actual - desired) / desired


def robustness(actual, desired, threshold):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        actual (array): Actual achieved output
        desired (array): Desired output at corresponding times to actual
        threshold (float): power specification as a percent deviation from profile
    """
    # if we're at a terminal node, we won't get power arrays, so return infinity (we're going to take a min of this rho and the path rho, so path rho will always get picked if we're at a terminal node)
    if type(actual) != list:
        actual = np.array([actual])
    if type(desired) != list:
        desired = np.array([desired])
    if len(actual) == 0:
        return np.inf
    percent_diffs = abs(np.array(actual) - np.array(desired)) / np.array(desired)
    rho = np.min(threshold - percent_diffs)
    return rho
