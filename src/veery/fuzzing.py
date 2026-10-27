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
            Do (function): Observation (sensor) disturbance distribution
            Da (function): Agent disturbance distribution
            Ds (function): State (environment) distribution
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
