"""
This code was modified from that created for the following publication:

Leo Tunkle, Kamal Abdulraheem, Linyu Lin, Majdi I. Radaideh,
Nuclear microreactor transient and load-following control with deep reinforcement learning,
Energy Conversion and Management: X,
Volume 27,
2025,
101090,
ISSN 2590-1745,
https://doi.org/10.1016/j.ecmx.2025.101090.

Tunkle et al.'s original codebase available at: https://github.com/aims-umich/microdrum-marl.
"""

import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class HolosPK:
    """Class representing the Holos Quad physical system using point kinetics equations."""

    def __init__(self):
        ##############################
        # Default reactor parameters #
        ##############################

        # values mostly from table 2 inChoi 2020
        self.sigma_xe = 2.65e-22  # xenon micro xsec m^2 according to online sources
        self.yield_i = 0.061  # yield iodine
        self.yield_xe = 0.002  # yield xenon
        self.lambda_xe = 2.09e-5  # decay of xenon s^-1
        self.lambda_i = 2.87e-5  # decay of iodine s^-1
        self.Sigma_f = 0.1117  # macro xsec fission m^-1, worked out backwards from estimated values in Choi 2020 p. 28 Fig.15c
        self.therm_n_vel = 2.19e3  # thermal neutron velocity m/s according to Wikipedia on neutron temp (0.25 eV)
        self.neutron_lifetime = 1.68e-3  # s, Lambda in paper
        self.beta = 0.004801
        self.betas = np.array(
            [
                1.42481e-04,
                9.24281e-04,
                7.79956e-04,
                2.06583e-03,
                6.71175e-04,
                2.17806e-04,
            ]
        )
        self.lambdas = np.array(
            [1.272e-02, 3.174e-02, 1.160e-01, 3.110e-01, 1.400e00, 3.870e00]
        )
        self.cp_f = 977  # specific heat of fuel
        self.cp_m = 1697  # specific heat of moderator
        self.cp_c = 5188.6  # specific heat of coolant
        self.M_f = 2002  # mass of fuel
        self.M_m = 11573  # mass of moderator
        self.M_c = 500  # mass of coolant
        self.heat_f = 0.96  # q in paper, fraction of heat deposited in fuel
        self.Tf0 = 832.4  # fuel initial temp, MPACT paper in K
        self.Tm0 = 830.22  # moderator initial temp, MPACT paper in K
        self.T_in = (
            795.47  # inlet temp, computed to make initial conditions at steady state
        )
        self.T_out = 1106  # outlet temp, sooyoung's code
        self.Tc0 = 814.35  # initial cladding temp, computed to make initial conditions at steady state
        self.K_fm = 1.17e6  # W/K heat transfer coefficient fuel to moderator
        self.K_mc = 2.16e5  # W/K heat transfer coefficient moderator to coolant
        self.M_dot = 17.5  # kg/s, mass flow rate
        self.alpha_f = -2.875e-5  # K^-1
        self.alpha_m = -3.696e-5  # K^-1
        self.alpha_c = 0.0  # K^-1
        self.n_0 = 2.25e13  # m^-3
        self.P_r = 22e6  # rated power in Watts
        self.u0 = 77.8  # degrees, steady state full power drum angle (77.56 earlier)
        self.rho_max = 0.00510  # max reactivity per drum (510 pcm)
        # steady state drum reactivity
        self.rho_ss = self.rho_max * (1 - np.cos(np.deg2rad(self.u0))) / 2
        # initial iodine concentration
        self.i0 = (
            self.yield_i * self.Sigma_f * self.therm_n_vel * self.n_0 / self.lambda_i
        )
        # initial xenon concentration
        self.xe0 = (
            self.yield_xe * self.Sigma_f * self.therm_n_vel * self.n_0
            + self.lambda_i * self.i0
        ) / (self.lambda_xe + self.sigma_xe * self.therm_n_vel * self.n_0)

    def get_initial_conditions(self):
        """Gets initial conditions using reactor params from init.

        Returns:
            list: state variables for initial condition of the reactor
        """
        n_r = 1  # neutron density
        c1, c2, c3, c4, c5, c6 = [n_r] * 6  # precursor concentrations
        Tf = self.Tf0  # fuel temp
        Tm = self.Tm0  # moderator temp
        Tc = self.Tc0  # coolant temp
        xe = self.xe0  # xenon concentration
        i = self.i0  # iodine concentration

        return [n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, xe, i]

    def calc_reactivity(self, state, drum_angles):
        """Calculates reactivity at a given state with given drum angles

        Args:
            state (tuple): tuple of reactor state variables in the form from get_initial_state()
            drum_angles (tuple): tuple of eight drum angles (in named order) in degrees

        Returns:
            float: reactivity value, rho
        """
        _, _, _, _, _, _, _, Tf, Tm, _, xe, _ = state

        drum_reactivity = np.sum(
            self.rho_max * (1 - np.cos(np.deg2rad(drum_angles))) / 2 - self.rho_ss
        )

        rho = (
            drum_reactivity
            + self.alpha_f * (Tf - self.Tf0)
            + self.alpha_m * (Tm - self.Tm0)
            - self.sigma_xe * (xe - self.xe0) / self.Sigma_f
        )

        return rho

    def drum_forcing(self, drum_angles, drum_action, time=1):
        """Creates a list of interpolator functions for drum angles over a time step determined by time.

        Args:
            drum_angles (list): list of current drum angles, in degrees
            drum_action (list): list of drum angle changes, in degrees
            time (int, optional): duration of a single control step in seconds. Defaults to 1 second.

        Returns:
            list: a list of interpolator functions (functions of time)
        """
        drum_forcers = []
        for i, drum_angle in enumerate(drum_angles):
            new_angle = np.clip(
                drum_angle + drum_action[i], 0, 180
            ).item()  # can't go beyond limits
            drum_forcers.append(interp1d([0, time], [drum_angle, new_angle]))

        assert len(drum_forcers) == len(drum_angles)  # sanity check
        return drum_forcers

    def reactor_dae(self, t, state, *drum_forcers):
        """Calculates derivatives for reactor state update.

        Args:
            t (float): time
            state (tuple): current (at time t) state vector for the reactor
            drum_forcers (list): list of drum forcer interpolated functions (in order)

        Returns:
            list: state derivatives
        """
        n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, xe, i = state
        # get the updated drum angles after action is taken using interpolated functions
        drum_angles = np.array([f(t) for f in drum_forcers])
        rho = self.calc_reactivity(state, drum_angles)
        precursor_concentrations = np.array([c1, c2, c3, c4, c5, c6])

        # Kinetics equations with six-delayed neutron groups
        # get d(neutron density)
        d_n_r = (
            (rho - self.beta) * n_r + np.sum(self.betas * precursor_concentrations)
        ) / self.neutron_lifetime

        # get d(precursor concentrations)[1-6]
        d_c1, d_c2, d_c3, d_c4, d_c5, d_c6 = (
            self.lambdas * n_r - self.lambdas * precursor_concentrations
        )

        # Thermal–hydraulics model of the reactor core
        # get d(fuel temp)
        d_Tf = (self.heat_f * self.P_r * n_r - self.K_fm * (Tf - Tc)) / (
            self.M_f * self.cp_f
        )

        # get d(moderator temp)
        d_Tm = (
            (1 - self.heat_f) * self.P_r * n_r
            + self.K_fm * (Tf - Tm)
            - self.K_mc * (Tm - Tc)
        ) / (self.M_m * self.cp_m)

        # get d(coolant temp)
        d_Tc = (
            self.K_mc * (Tm - Tc) - 2 * self.M_dot * self.cp_c * (Tc - self.T_in)
        ) / (self.M_c * self.cp_c)

        # Xenon and Iodine dynamics
        n_rate_density = self.therm_n_vel * self.n_0 * n_r
        # get d(iodine concentration)
        d_i = self.yield_i * self.Sigma_f * n_rate_density - self.lambda_i * i
        # get d(xenon concentration)
        d_xe = (
            self.yield_xe * self.Sigma_f * n_rate_density
            + self.lambda_i * i
            - self.lambda_xe * xe
            - self.sigma_xe * xe * n_rate_density
        )

        return [d_n_r, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6, d_Tf, d_Tm, d_Tc, d_xe, d_i]
