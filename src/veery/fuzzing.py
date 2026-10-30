from scipy.stats import norm
import numpy as np


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

    def sample(self, state, action):
        return Disturbance(
            self.Do.sample(state), self.Da.sample(action), self.Ds.sample(state, action)
        )

    def logpdf(self, x):
        # gets the log-likelihood for the disturbance, depending on the disturbance distribution
        return self.Do.logpdf(x.xo), self.Da.logpdf(x.xa), self.Ds.logpdf(x.xs)
