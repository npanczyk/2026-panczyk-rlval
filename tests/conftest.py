"""Dummy bakery system for testing veery.mcts without a real environment.

Written by Nataly, with fixes suggested by Claude (Opus 5.5).
"""

import itertools

import pytest
from veery.fuzzing import *
from veery.accessories import robustness


class DummyModel:
    def predict(self, observation=None, deterministic=True):
        # the baker always tries to bake one cake per step
        return (1, None)


class DummyDisturbanceDistribution:
    def __init__(self, xs_values=(0,)):
        # cycle through xs_values so tests can script which disturbance each sample gives
        self.xs_values = itertools.cycle(xs_values)

    def sample(self, state=None, action=None):
        return Disturbance(0.1, 0, next(self.xs_values))

    def logpdf(self, x):
        # mcts only reads the observation (0th) entry, so put the cost of a big xs there; this makes larger disturbances less likely and gives rank_failures something to sort on
        return (-2.0 - abs(x.xs), 0, 0)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_disturbance_dist():
    return DummyDisturbanceDistribution()


@pytest.fixture
def failing_disturbance_dist():
    # alternates nominal and one extra cake per step, so some branches break the order
    return DummyDisturbanceDistribution(xs_values=(0, 1))


class Bakery:
    def __init__(self):
        self.runtime = 5
        self.reset()

    def profile(self, time):
        # the order: one cake per hour
        return time

    def reset(self):
        self.time = 0
        self.flour = 10
        self.sugar = 5
        self.eggs = 10
        self.cakes = 0
        self._update_state()

    def _update_state(self):
        self.state = {
            "flour": self.flour,
            "sugar": self.sugar,
            "eggs": self.eggs,
            "time": self.time,
            "cakes": self.cakes,
        }

    def step(self, action=0, disturbance=None):
        # the state disturbance xs is extra (or ruined, if negative) cakes on top of the action
        xs = disturbance.xs if disturbance is not None else 0
        made = action + xs
        self.time += 1
        self.cakes += made
        self.flour -= 2 * made
        self.sugar -= 1 * made
        self.eggs -= 2 * made
        self._update_state()
        return self.cakes, None, False, False, {}

    def state_robustness(self, threshold, percent=False):
        # same spec as the rollouts: cakes stay within threshold of the order
        return robustness(
            self.cakes, self.profile(self.time), threshold=threshold, percent=percent
        )

    def get_snapshot(self):
        return {
            "flour": self.flour,
            "sugar": self.sugar,
            "eggs": self.eggs,
            "time": self.time,
            "cakes": self.cakes,
            "runtime": self.runtime,
            "observation": self.cakes,
        }

    def set_snapshot(self, snap):
        self.time = snap["time"]
        self.flour = snap["flour"]
        self.sugar = snap["sugar"]
        self.eggs = snap["eggs"]
        self.cakes = snap["cakes"]
        self._update_state()

    def rollout_from_snapshot(self, model, snap, disturbance_distribution):
        self.set_snapshot(snap)
        actual_cakes = []
        desired_cakes = []
        while self.time < self.runtime:
            action, _ = model.predict(observation=self.cakes, deterministic=True)
            x = disturbance_distribution.sample(state=self.state, action=action)
            self.step(action=action, disturbance=x)
            actual_cakes.append(self.cakes)
            desired_cakes.append(self.profile(self.time))
        return actual_cakes, desired_cakes


@pytest.fixture
def bakery():
    return Bakery()
