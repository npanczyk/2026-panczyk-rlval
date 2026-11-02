import numpy as np
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def clamp(number, min, max):
    if number < min:
        return min
    elif number > max:
        return max
    else:
        return number


# this system is based on the Algorithm's for Validation (Kochenderfer et al.) example case!
class Disturbance:
    def __init__(self, xo, xa, xs):
        self.xo = xo
        self.xa = xa
        self.xs = xs
        pass


class DisturbanceDistribution:
    def __init__(self, Do, Da, Ds):
        self.Do = Do
        self.Da = Da
        self.Ds = Ds
        pass


def plot_observation_disturbances(xs, sigma):
    fig, ax = plt.subplots()
    for x in xs:
        ax.scatter(x.xo[0, 0], x.xo[1, 0], color="orange")
    ax.set_xlabel(r"xo, $\theta$")
    ax.set_ylabel(r"xo, $\omega$")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.savefig(f"../../results/disturbance_plot_{sigma}.svg", dpi=300)
    return ax


def plot_trajectories(theta_matrix, sigma):
    fig, ax = plt.subplots()
    ax.plot(theta_matrix, color="steelblue", alpha=0.5, linewidth=0.8)
    plt.axhline(np.pi / 4, color="red", linewidth=1)
    plt.axhline(-np.pi / 4, color="red", linewidth=1)
    ax.set_title(r"$\sigma = $ " + f"{sigma}")
    plt.xlabel("t, [s]")
    plt.ylabel(r"$\theta$, [rad]")
    plt.savefig(f"../../results/inverted_pendulum_{sigma}.svg", dpi=300)
    return ax


class InvertedPendulum:
    def __init__(self, m, l, dt):
        self.m = m  # mass of the pendulum
        self.l = l  # length of the pendulum
        self.dt = dt  # time step in seconds
        self.g = 9.81  # m/s**2, gravitational constant
        self.w_max = 8.0  # maximum angular velocity
        self.a_max = 2.0  # maximum applicable torque
        self.alpha = np.array(
            [[-15, -8]]
        )  # (1 x 2) gain matrix for proportional control
        # define a constant profile so that we can calculate robustness
        self.profile = lambda t: 0

    def reset(self):
        self.time = 0
        theta0 = np.random.uniform(-np.pi / 16, np.pi / 16)
        w0 = np.random.uniform(-1.0, 1.0)
        xo = {"theta": 0, "omega": 0}
        observation = self._get_observation(xo, reset=True)
        self._log_history(observation=observation, init=True)
        self._last_observation = observation
        return observation

    def _get_observation(self, xo):
        # additive noise sensor
        # grab disturbance distribution from init and sample
        # s = np.array([[s[0]], [s[1]]])
        # return s + xo
        return {"theta": self._theta + xo["theta"], "omega": self._omega + xo["omega"]}

    def _log_history(self, observation, init=False):
        if init:
            self.history = [
                [
                    self.time,
                    observation["theta"].item(),  # observed angle
                    self.profile(self.time),  # desired angle
                    observation["omega"].item(),  # observed angular velocity
                    *self.state,  # includes actual power
                ]
            ]
            return

    def get_action(self, o, xa):
        # proportional counter
        # take the dot product of the gain matrix and the observation
        torque = self.alpha @ o
        return torque[0][0] + xa[0]

    def get_state(self, s, a, xs):
        """Gets values for the next state.

        Args:
            s (tuple): theta, w - current angle and angular velocity
            a (_type_): action, proposed torque from controller
        Returns:
            theta, omega: updated state variables
        """
        theta, w = s[0], s[1]
        dt, g, m, l = self.dt, self.g, self.m, self.l
        a = clamp(a, -1 * self.a_max, self.a_max)
        w = w + ((3 * g / (2 * l)) * np.sin(theta) + (3 * a) / (m * l**2)) * dt
        theta = theta + w * dt
        w = clamp(w, -1 * self.w_max, self.w_max)
        return theta + xs[0, 0], w + xs[1, 0]

    def step(self, action, disturbance):
        self.time += self.dt
        xo = disturbance_dist.Do.rvs(size=(2, 1))
        o = self.get_observation(s, xo)
        xa = disturbance_dist.Da.rvs(size=1)
        a = self.get_action(o, xa)
        xs = disturbance_dist.Ds.rvs(size=(2, 1))
        sprime = self.get_state(s, a, xs)
        return o, a, sprime, Disturbance(xo, xa, xs)

    def rollout(self, runtime, disturbance_dist):
        s = self.get_initial_state()  # initial state: (theta, omega)
        time = np.arange(0, runtime, self.dt)
        thetas = np.zeros(len(time))  # empty array of angles from centerline
        omegas = np.zeros(len(time))  # empty array of angular velocities
        xs = []
        thetas[0] = s[0]
        omegas[0] = s[1]
        for i, t in enumerate(time):
            o, a, s, x = self.step(s, disturbance_dist)
            thetas[i] = s[0]
            omegas[i] = s[1]
            xs.append(x)
        return thetas, omegas, xs

    def check_spec(self, trajectory):
        """Checks if a trajectory is within spec

        Args:
            trajectory (Numpy Array): array of theta values over course of rollout

        Returns:
            bool: True if all elements of the trajectory are within spec
        """
        return (np.abs(trajectory) <= np.pi / 4).all()


if __name__ == "__main__":
    n_rollouts = 300
    sig = 0.15
    disturbance_dist = DisturbanceDistribution(
        Do=norm(0, sig), Da=norm(0, 0), Ds=norm(0, 0)
    )
    pendulum = InvertedPendulum(m=1.0, l=1.0, dt=0.05)
    runtime = 10  # length of rollout in seconds
    timesteps = int(
        np.ceil(runtime / pendulum.dt)
    )  # this formula has edge cases, may need to change later to linspace in rollouts method
    successes = np.zeros(n_rollouts)
    theta_matrix = np.zeros((timesteps, n_rollouts))

    for r in range(n_rollouts):
        thetas, omegas, xs = pendulum.rollout(runtime, disturbance_dist)
        theta_matrix[:, r] = thetas
        successes[r] = pendulum.check_spec(thetas)

    plot_observation_disturbances(xs, sigma=sig)
    plot_trajectories(theta_matrix, sigma=sig)
    print(f"Success Rate: {successes.sum()/n_rollouts}")
