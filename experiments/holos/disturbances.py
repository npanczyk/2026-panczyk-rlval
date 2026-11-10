from scipy.stats import norm
import numpy as np
from veery.fuzzing import *


class Do(ComponentDistribution):
    "Observation distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p) and drum angle (drum_angles). Drum angle fuzz should correspond to real physical space (0, 180 degrees). Power should correspond to fractional physical space (0, 1). Change in power (dp) is fuzzed automatically because it is calculated using fuzzy power values."

    def __init__(self, sigma_p=0.01, sigma_drum=0.005):
        super().__init__(
            {
                "p": DisturbedVariable(gaussian(sigma_p)),
                "drum_angles": DisturbedVariable(gaussian(sigma_drum), size=8),
            }
        )


class Da(ComponentDistribution):
    "Action distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for drum angle changes (drum_angles). Noise for action space should correspond to real physical space (-0.5 deg/s, 0.5 deg/s)."

    def __init__(self, sigma_dtheta=0):
        super().__init__({"dtheta": DisturbedVariable(gaussian(sigma_dtheta), size=8)})


class Ds(ComponentDistribution):
    "State disturbance distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p), change in power (dp), and drum angle (drum_angles)."

    def __init__(
        self,
        sigma_nr=0,  # neutron density
        sigma_precursors=0,  # precursor concentration
        sigma_Tf=0,  # fuel temperature
        sigma_Tm=0,  # moderator temperature
        sigma_Tc=0,  # coolant temperature
        sigma_xe=0,  # xenon concentration
        sigma_i=0,  # iodine concentration
    ):
        super().__init__(
            {
                "nr": DisturbedVariable(gaussian(sigma_nr)),
                "precursors": DisturbedVariable(gaussian(sigma_precursors), size=6),
                "Tf": DisturbedVariable(gaussian(sigma_Tf)),
                "Tm": DisturbedVariable(gaussian(sigma_Tm)),
                "Tc": DisturbedVariable(gaussian(sigma_Tc)),
                "xe": DisturbedVariable(gaussian(sigma_xe)),
                "i": DisturbedVariable(gaussian(sigma_i)),
            }
        )
