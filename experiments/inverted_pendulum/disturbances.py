from scipy.stats import norm
import numpy as np
from veery.fuzzing import *


class Do(ComponentDistribution):
    """Observation disturbance class for the inverted pendulum system."""

    def __init__(self, sigma_theta, sigma_omega):
        """Sets the standard deviation and defines the disturbance distribution forms for the inverted pendulum system.

        Args:
            sigma_theta (float): standard deviation for the observation disturbances for the pendulum angle
            sigma_omega (float): standard deviation for the
        """
        super().__init__(
            {
                "theta": DisturbedVariable(gaussian(sigma_theta)),
                "omega": DisturbedVariable(gaussian(sigma_omega)),
            }
        )


class Da(ComponentDistribution):
    """Action disturbance class for the inverted pendulum system."""

    def __init__(self, sigma_torque):
        super().__init__({"torque": DisturbedVariable(gaussian(sigma_torque))})


class Ds(ComponentDistribution):
    """State disturbance distribution for the inverted pendulum system."""

    def __init__(self, sigma_theta=0, sigma_omega=0):
        """
        Args:
            sigma_theta (float): Variance for a normal distribution of noise for the actual angle of the pendulum (should pretty much always be zero)
            sigma_omega (float): Variance for a normal distribution of noise for the actual angular velocity of the pendulum (like the angle, this should also pretty much always be zero)
        """
        super().__init__(
            {
                "theta": DisturbedVariable(gaussian(sigma_theta)),
                "omega": DisturbedVariable(gaussian(sigma_omega)),
            }
        )
