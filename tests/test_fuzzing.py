import pytest
from veery.fuzzing import *
import numpy as np


def test_zerodist_sampling():
    X = ZeroDist()
    assert X.rvs() == float(0)
    assert X.rvs(2).shape == (2,)


def test_zerodist_logpdf():
    X = ZeroDist()
    assert np.all(X.logpdf(0.92) == 0)


def test_gaussian():
    X = gaussian(0.01, 0)
    assert X.mean() == 0
    assert X.std() == 0.01


def test_zero_gaussian():
    X = gaussian(0, 0)
    assert type(X) == ZeroDist
    Y = gaussian(0, 100)
    assert type(Y) == ZeroDist


@pytest.fixture
def angles():
    return DisturbedVariable(gaussian(0.01, 0), 8)


@pytest.fixture
def power():
    return DisturbedVariable(gaussian(0.05, 0), 1)


@pytest.fixture
def temp():
    return DisturbedVariable(ZeroDist(), 1)


@pytest.fixture
def obs_variables(angles, power):
    return {"angles": angles, "power": power}


@pytest.fixture
def Do(obs_variables):
    return ComponentDistribution(obs_variables)


@pytest.fixture
def Da(temp):
    return ComponentDistribution({"temp": temp})


@pytest.fixture
def Ds(temp):
    return ComponentDistribution({"temp": temp})


def test_DisturbedVariable(angles):
    assert angles.size == 8


def test_ComponentDistribution(Do):
    assert Do.sample()["angles"].shape == (8,)
    x = {"angles": np.zeros(8), "power": np.zeros(1)}
    assert type(Do.logpdf(x)) == np.float64


def test_DisturbanceDistribution(Do, Da, Ds):
    D = DisturbanceDistribution(Do, Da, Ds)
    x = D.sample()
    assert type(x) == Disturbance
    assert type(D.logpdf(x)) == tuple


@pytest.fixture(scope="session")
def save_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("results")
    return path


def test_plot_samples(Do, Da, Ds, save_path):
    D = DisturbanceDistribution(Do, Da, Ds)
    D.plot_samples(Do, "angles", 100, save_path)
    assert (save_path / "angles_sample_dist.png").exists()
