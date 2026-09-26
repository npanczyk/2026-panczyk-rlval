"""
Tests for fuzzing.py's Disturbance/DisturbanceDistribution/Do/Da/Ds classes.

The tests in this file were originally drafted by Claude Sonnet 5 but modified and verified by Nataly.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from fuzzing import Disturbance, DisturbanceDistribution, Do, Da, Ds


def test_disturbance_stores_fields():
    # just checking the disturbance class structure with dummy values
    d = Disturbance(xo={"p": 1}, xa=np.ones(8), xs=np.zeros(12))
    assert d.xo == {"p": 1}
    np.testing.assert_array_equal(d.xa, np.ones(8))
    np.testing.assert_array_equal(d.xs, np.zeros(12))


# --- Do ---

def test_do_sample_keys_and_shapes():
    xo = Do(sigma_p=0.01, sigma_dp=0.01, sigma_drum=0.01).sample()
    assert set(xo.keys()) == {"p", "dp", "drum_angles"}
    assert np.shape(xo["drum_angles"]) == (8,)
    assert np.isscalar(xo["p"]) or np.shape(xo["p"]) == ()
    assert np.isscalar(xo["dp"]) or np.shape(xo["dp"]) == ()


def test_do_sample_zero_sigma_is_exactly_zero():
    xo = Do(sigma_p=0, sigma_dp=0, sigma_drum=0).sample()
    assert xo["p"] == 0
    assert xo["dp"] == 0
    np.testing.assert_array_equal(xo["drum_angles"], np.zeros(8))


# --- Da ---

def test_da_sample_shape():
    xa = Da(sigma_dtheta=0.01).sample()
    assert xa.shape == (8,)


def test_da_sample_zero_sigma_is_exactly_zero():
    xa = Da(sigma_dtheta=0).sample()
    np.testing.assert_array_equal(xa, np.zeros(8))


# --- Ds ---

def test_ds_sample_array_shape_and_zero_default():
    xs = Ds().sample(return_format="array")
    assert xs.shape == (12,)
    np.testing.assert_array_equal(xs, np.zeros(12))


def test_ds_sample_dict_keys_and_shapes():
    xs = Ds().sample(return_format="dict")
    assert set(xs.keys()) == {"nr", "precursors", "Tf", "Tm", "Tc", "xe", "i"}
    assert np.shape(xs["precursors"]) == (6,)


def test_ds_sample_array_matches_state_ordering():
    # [n_r, c1..c6, Tf, Tm, Tc, xe, i] -- nonzero only on Tf/Tm/Tc (indices 7,8,9)
    xs = Ds(sigma_Tf=1, sigma_Tm=1, sigma_Tc=1).sample(return_format="array")
    nonzero_idx = np.flatnonzero(xs)
    assert set(nonzero_idx).issubset({7, 8, 9})


# --- DisturbanceDistribution wiring ---

def test_disturbance_distribution_sample_calls_each_component_correctly():
    mock_Do, mock_Da, mock_Ds = MagicMock(), MagicMock(), MagicMock()
    mock_Do.sample.return_value = {"p": 0.1}
    mock_Da.sample.return_value = np.ones(8)
    mock_Ds.sample.return_value = np.zeros(12)

    dist = DisturbanceDistribution(Do=mock_Do, Da=mock_Da, Ds=mock_Ds)
    state, action = np.zeros(12), np.zeros(8)
    x = dist.sample(state, action)

    mock_Do.sample.assert_called_once_with(state)
    mock_Da.sample.assert_called_once_with(action)
    mock_Ds.sample.assert_called_once_with(state, action)
    assert x.xo == {"p": 0.1}
    np.testing.assert_array_equal(x.xa, np.ones(8))
    np.testing.assert_array_equal(x.xs, np.zeros(12))