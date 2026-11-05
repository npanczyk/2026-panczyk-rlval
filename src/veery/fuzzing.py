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
        """Plots N samples for a given distribution and variable of interest.

        Args:
            d_type (Disturbance and type): One of disturbance.Do, disturbance.Da, or disturbance.Ds
            variable (str): Variable name from the disturbance of interest (application dependent)
            n_samples (int, optional): Number of samples to take. Defaults to 1000.
            save_path (str, optional): Folder path where the figure should save to. Defaults to None and just shows the figure.
        """
        fig, ax = plt.subplots()
        samples = [dist_type.sample()[variable] for i in range(n_samples)]
        ax.hist(samples, 100)
        if save_path:
            Path(save_path).mkdir(exist_ok=True)
            fig.savefig(f"{save_path}/sample_dist.png", dpi=300)
        else:
            fig.show()
