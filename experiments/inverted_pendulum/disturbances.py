from scipy.stats import norm
import numpy as np
from veery.fuzzing import ZeroDist


class Do:
    """Observation disturbance class for the inverted pendulum system."""

    def __init__(self, sigma_theta, sigma_omega):
        """Sets the standard deviation and defines the disturbance distribution forms for the inverted pendulum system.

        Args:
            sigma_theta (float): standard deviation for the observation disturbances for the pendulum angle
            sigma_omega (float): standard deviation for the
        """
        self.sigma_theta = sigma_theta
        self.sigma_omega = sigma_omega
        if sigma_theta == 0:
            self.theta_dd = ZeroDist()
        else:
            self.theta_dd = norm(0, sigma_theta)
        if sigma_omega == 0:
            self.omega_dd = ZeroDist()
        else:
            self.omega_dd = norm(0, sigma_omega)

    def sample(self, state=None):
        """Samples observaton noise following the previously defined distributions.

        Args:
            state (array-like, optional): State vector, required if noise sample depends on current state. Otherwise, defaults to None.
        """
        return {"theta": self.theta_dd.rvs(), "omega": self.omega_dd.rvs()}

    def logpdf(self, xo):
        """Gets the loglikelihood of an observation noise sample based on the defined distributions.

        Args:
            xo (dict): Observation noise dictionary in the same format as sample()
        """
        return self.theta_dd.logpdf(xo["theta"]) + self.omega_dd.logpdf(xo["omega"])


class Da:
    """Action disturbance class for the inverted pendulum system."""

    def __init__(self, sigma_torque):
        self.sigma_torque = sigma_torque
        if sigma_torque == 0:
            self.torque_dd = ZeroDist()
        else:
            self.torque_dd = norm(0, sigma_torque)

    def sample(self, observation=None):
        """Gets randomly sampled noise for the applied action (torque).

        Args:
            observation (dict, optional): Current observation dictionary. Required if action noise depends on current obseration. Defaults to None.

        Returns:
            dict: Dictionary containing the random noise to be applied to the torque action. Keyed by "torque"
        """
        return {"torque": self.torque_dd.rvs()}

    def logpdf(self, xa):
        """Computes the log-likelihood of an action disturbance

        Args:
            xa (dict): Action disturbance with key "torque"

        Returns:
            float: log-likelihood value
        """
        return self.torque_dd.logpdf(xa["torque"])


class Ds:
    """State disturbance distribution for the inverted pendulum system."""

    def __init__(self, sigma_theta=0, sigma_omega=0):
        """
        Args:
            sigma_theta (float): Variance for a normal distribution of noise for the actual angle of the pendulum (should pretty much always be zero)
            sigma_omega (float): Variance for a normal distribution of noise for the actual angular velocity of the pendulum (like the angle, this should also pretty much always be zero)
        """
        self.sigma_theta = sigma_theta
        self.sigma_omega = sigma_omega
        if sigma_theta == 0:
            self.theta_dd = ZeroDist()
        else:
            self.theta_dd = norm(0, sigma_theta)
        if sigma_omega == 0:
            self.omega_dd = ZeroDist()
        else:
            self.omega_dd = norm(0, sigma_omega)

    def sample(self, state=None, action=None):
        """Samples disturbances for the state disturbances (noise within the environment itself).

        Args:
            state (array-like, optional): The current state, required if it impacts the state noise distribution. Defaults to None.
            action (array-like, optional): The prescribed action, required if it impacts the state noise distribution. Defaults to None.
        """
        return {"theta": self.theta_dd.rvs(), "omega": self.omega_dd.rvs()}

    def logpdf(self, xs):
        """Gets the log-likelihood of state disturbance

        Args:
            xs (dict): state disturbance dictionary, following the format of Ds.sample()
        """
        return self.theta_dd.logpdf(xs["theta"]) + self.omega_dd.logpdf(xs["omega"])
