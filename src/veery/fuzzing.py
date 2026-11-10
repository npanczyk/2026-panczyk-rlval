from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ZeroDist:
    def __init__(self):
        return

    def rvs(self, size=None):
        """Samples from a "zero distribution" (deterministic)

        Args:
            size (int, optional): Number of samples to be drawn. Defaults to None.

        Returns:
            Numpy array or float: Random samples
        """
        if size:
            return np.zeros(size)
        else:
            return 0.0

    def logpdf(self, x):
        """Gets the loglikelihood given a sample and returns 0 for deterministic runs.

        Args:
            x (array_like): The disturbance value/sample you want the loglikelihood of

        Returns:
            array: Zero array
        """
        return np.zeros(1)


def gaussian(sigma, mean=0):
    """Wrapper for normal distribution that handles the zero distribution (when sigma is 0).

    Args:
        sigma (float): Standard deviation of the distribution
        mean (int, optional): Mean of the distribution. Defaults to 0.

    Returns:
        distribution object: either a ZeroDist() object or scipy.stats.norm instance
    """
    if sigma == 0:
        return ZeroDist()
    else:
        return norm(mean, sigma)


class DisturbedVariable:
    """Sets the template for a disturbance variable that contains that variable's distribution (gaussian, uniform, etc.) and its size"""

    def __init__(self, distribution, size=None):
        self.distribution = distribution
        self.size = size


class ComponentDistribution:
    """Disturbance distribution class for Do, Da, or Ds, each containing independent named variables from the Variable class."""

    def __init__(self, variables_dict):
        self.variables = variables_dict  # dictionary of the form: {name: variable, name: variable, etc.}

    def sample(self, *dependencies):
        """Gets randomly generated samples for a disturbance distribution.
        Note: dependencies are state/action/observation information that can be overriden by a subclass-specific sample() method

        Returns:
            dict: keys are the names of the variables, values are the samples
        """
        return {
            name: var.distribution.rvs(size=var.size)
            for name, var in self.variables.items()
        }

    def logpdf(self, x):
        """Gets the logpdf for a disturbance.

        Args:
            x (dict): dictionary where the keys are the names of the variables and the values are the disturbances in the same format as sample() outputs
        Returns:
            float: the sum of the logpdf of each variable's disturbance
        """
        loglikelihood = 0
        for name, var in self.variables.items():
            loglikelihood += np.sum(var.distribution.logpdf(x[name]))
        return loglikelihood


class Disturbance:
    def __init__(self, xo, xa, xs):
        """
        Args:
            xo (float): Observation (sensor) disturbance
            xa (float): Agent disturbance
            xs (float): State (environment) disturbance
        """
        self.xo = xo
        self.xa = xa
        self.xs = xs
        pass


class DisturbanceDistribution:
    def __init__(self, Do, Da, Ds):
        """
        Args:
            Do (class): Observation (sensor) disturbance distribution. Must be a class that has a sample() method that returns a dictionary with disturbed quantities (str) as keys and disturbance value (float) as values and a logpdf() method that returns a float.
            Da (class): Agent disturbance distribution. Must be a class that has a sample() method that returns a dictionary with disturbed quantities as keys (str) and disturbance value (float) as values and a logpdf() method that returns a float.
            Ds (class): State (environment) distribution. Must be a class that has a sample() method that returns a dictionary with disturbed quantities as keys (str) and disturbance value (float) as values and a logpdf() method that returns a float.
        """
        self.Do = Do
        self.Da = Da
        self.Ds = Ds

    def sample(self, state=None, action=None):
        return Disturbance(
            self.Do.sample(state), self.Da.sample(action), self.Ds.sample(state, action)
        )

    def logpdf(self, x):
        # gets the log-likelihood for the disturbance, depending on the disturbance distribution
        return self.Do.logpdf(x.xo), self.Da.logpdf(x.xa), self.Ds.logpdf(x.xs)

    def plot_samples(self, dist_type, variable, n_samples=1000, save_path=None):
        """Plots a histogram of N samples for one variable of one component distribution.

        Args:
            dist_type (ComponentDistribution): One of self.Do, self.Da, or self.Ds
            variable (str): Variable name from the component distribution of interest (application dependent)
            n_samples (int, optional): Number of samples to take. Defaults to 1000.
            save_path (str, optional): Folder path where the figure should save to. Defaults to None and just shows the figure.
        """
        fig, ax = plt.subplots()
        # ravel so vector variables (e.g. 8 drum angles) are pooled into one histogram
        samples = np.concatenate(
            [np.ravel(dist_type.sample()[variable]) for i in range(n_samples)]
        )
        ax.hist(samples, 100)
        ax.set_xlabel(f"{variable} disturbance")
        ax.set_ylabel("Count")
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            # name the file by variable so plotting several variables doesn't overwrite one file
            fig.savefig(f"{save_path}/{variable}_sample_dist.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()
