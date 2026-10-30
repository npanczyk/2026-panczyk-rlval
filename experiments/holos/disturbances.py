from scipy.stats import norm
import numpy as np


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
        return {"p": self.power_dd.rvs(), "drum_angles": self.drum_dd.rvs(size=8)}

    def logpdf(self, xo):
        """Gets the log-likelihood of an observation disturbance

        Args:
            xo (Do object): Observation disturbance to quantify the log likelihood for

        Returns:
            float: log likelihood value
        """
        l = self.power_dd.logpdf(xo["p"]) + self.drum_dd.logpdf(xo["drum_angles"]).sum()
        return l


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
        return {"dtheta": self.dtheta_dd.rvs(size=8)}

    def logpdf(self, xa):
        """Gets the log-likelihood of an observation disturbance

        Args:
            xa (Da object): Action disturbance to quantify the log likelihood for

        Returns:
            float: log likelihood value
        """
        return self.dtheta_dd.logpdf(xa["dtheta"]).sum()


class Ds:
    "State distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p), change in power (dp), and drum angle (drum_angles)."

    def __init__(
        self,
        sigma_nr=0.00001,
        sigma_precursors=0.00001,
        sigma_Tf=0.00001,
        sigma_Tm=0.00001,
        sigma_Tc=0.00001,
        sigma_xe=0.00001,
        sigma_i=0.00001,
    ):
        # initialize the disturbance distributions for the state variables
        self.nr_dd = norm(0, sigma_nr)  # neutron density dd
        self.precursor_dd = norm(0, sigma_precursors)  # precursor concentration dd
        self.Tf_dd = norm(0, sigma_Tf)  # fuel temperature
        self.Tm_dd = norm(0, sigma_Tm)  # moderator temperature
        self.Tc_dd = norm(0, sigma_Tc)  # coolant temperature
        self.xe_dd = norm(0, sigma_xe)  # xenon concentration
        self.i_dd = norm(0, sigma_i)  # iodine concentration

        # STATE VARIABLES FOR REFERENCE
        # n_r = 1  # neutron density
        # c1, c2, c3, c4, c5, c6 = [n_r] * 6  # precursor concentrations
        # Tf = self.Tf0  # fuel temp
        # Tm = self.Tm0  # moderator temp
        # Tc = self.Tc0  # coolant temp
        # xe = self.xe0  # xenon concentration
        # i = self.i0  # iodine concentration

    def sample(self, state=None, action=None):
        """Samples disturbances for one step of state computation.

        Args:
            state (numpy array, optional): The current state. Defaults to None.
            action (numpy array, optional): The current action. Defaults to None.
            return_format (str, optional): Whether the sample disturbance should be returned as a dict or an array. Defaults to "array".

        Returns:
            dict: set of disturbances across all state variables
        """

        return {
            "nr": self.nr_dd.rvs(),
            "precursors": self.precursor_dd.rvs(size=6),
            "Tf": self.Tf_dd.rvs(),
            "Tm": self.Tm_dd.rvs(),
            "Tc": self.Tc_dd.rvs(),
            "xe": self.xe_dd.rvs(),
            "i": self.i_dd.rvs(),
        }

    def logpdf(self, xs):
        llnr = self.nr_dd.logpdf(xs["nr"])
        llpc = self.precursor_dd.logpdf(xs["precursors"]).sum()
        lltf = self.Tf_dd.logpdf(xs["Tf"])
        lltm = self.Tm_dd.logpdf(xs["Tm"])
        lltc = self.Tc_dd.logpdf(xs["Tc"])
        llxe = self.xe_dd.logpdf(xs["xe"])
        lli = self.i_dd.logpdf(xs["i"])

        return llnr + llpc + lltf + lltm + lltc + llxe + lli
