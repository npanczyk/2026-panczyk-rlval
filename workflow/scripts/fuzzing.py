from scipy.stats import norm
import numpy as np
# do not import env here, you'll get a circular import

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
        return Disturbance(self.Do.sample(state), self.Da.sample(action), self.Ds.sample(state, action))

    def logpdf(self, x):
        # gets the log-likelihood for the disturbance, only considering observation disturbances for now
        return self.Do.logpdf(x.xo)

class Do:
    "Observation distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p) and drum angle (drum_angles). Drum angle fuzz should correspond to real physical space (0, 180 degrees). Power should correspond to fractional physical space (0, 1). Change in power (dp) is fuzzed automatically because it is calculated using fuzzy power values."
    def __init__(self, sigma_p=0.01, sigma_drum=0.005):
        # initialize the disturbance distributions for the observation variables
        # DO NOT disturb pnext (assume the controller reads the prescribed power correctly)
        self.power_dd = norm(0, sigma_p)
        self.drum_dd = norm(0, sigma_drum)

    def sample(self, state=None):
        """Gets an observation disturbance (adds sensor noise) for a single step

        Args:
            state (numpy array, optional): current system state vector. Defaults to None.

        Returns:
            dict: disturbances for three of four observation variables (pnext never gets fuzzed)
        """
        return {
            "p": self.power_dd.rvs(),
            "drum_angles": self.drum_dd.rvs(size=8)
        }

    def logpdf(self, xo):
        """Gets the log-likelihood of an observation disturbance

        Args:
            xo (Do object): Observation disturbance to quantify the log likelihood for

        Returns:
            float: log likelihood value
        """
        return self.power_dd.logpdf(xo["p"]) + self.drum_dd.logpdf(xo["drum_angles"]).sum()

class Da:
    "Action distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for drum angle changes (drum_angles). Noise for action space should correspond to real physical space (-0.5 deg/s, 0.5 deg/s)."
    def __init__(self, sigma_dtheta=0.001):
        # initialize the disturbance distributions for the action variables
        self.dtheta_dd = norm(0, sigma_dtheta)

    def sample(self, observation=None):
        """Gets an action disturbance list for all drums for a single step

        Args:
            observation (dict, optional): Current observation. Defaults to None.

        Returns:
            numpy array: 8 element array with disturbances for dtheta 
        """
        return self.dtheta_dd.rvs(size=8)

class Ds:
    "State distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p), change in power (dp), and drum angle (drum_angles)."
    def __init__(
            self, 
            sigma_nr=0, 
            sigma_precursors=0, 
            sigma_Tf=0, 
            sigma_Tm=0,
            sigma_Tc=0,
            sigma_xe=0,
            sigma_i=0):
        # initialize the disturbance distributions for the state variables
        self.nr_dd = norm(0, sigma_nr) # neutron density dd
        self.precursor_dd = norm(0, sigma_precursors) # precursor concentration dd
        self.Tf_dd = norm(0, sigma_Tf) # fuel temperature
        self.Tm_dd = norm(0, sigma_Tm) # moderator temperature
        self.Tc_dd = norm(0, sigma_Tc) # coolant temperature
        self.xe_dd = norm(0, sigma_xe) # xenon concentration
        self.i_dd = norm(0, sigma_i) # iodine concentration

        # STATE VARIABLES FOR REFERENCE
        # n_r = 1  # neutron density
        # c1, c2, c3, c4, c5, c6 = [n_r] * 6  # precursor concentrations
        # Tf = self.Tf0  # fuel temp
        # Tm = self.Tm0  # moderator temp
        # Tc = self.Tc0  # coolant temp
        # xe = self.xe0  # xenon concentration
        # i = self.i0  # iodine concentration

    def sample(self, state=None, action=None, return_format="array"):
        """Samples disturbances for one step of state computation. 

        Args:
            state (numpy array, optional): The current state. Defaults to None.
            action (numpy array, optional): The current action. Defaults to None.
            return_format (str, optional): Whether the sample disturbance should be returned as a dict or an array. Defaults to "array".

        Returns:
            either numpy array or dict: set of disturbances across all state variables
        """
        if return_format == "array":
            return np.concatenate([
                [self.nr_dd.rvs()],
                self.precursor_dd.rvs(size=6),
                [self.Tf_dd.rvs()],
                [self.Tm_dd.rvs()],
                [self.Tc_dd.rvs()],
                [self.xe_dd.rvs()],
                [self.i_dd.rvs()],
            ])

        elif return_format == "dict":
            return  {
                        "nr" : self.nr_dd.rvs(),
                        "precursors": self.precursor_dd.rvs(size=6),
                        "Tf": self.Tf_dd.rvs(),
                        "Tm": self.Tm_dd.rvs(),
                        "Tc": self.Tc_dd.rvs(),
                        "xe": self.xe_dd.rvs(),
                        "i": self.i_dd.rvs() 
                    }
