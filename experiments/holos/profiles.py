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

from scipy.interpolate import interp1d
import helpers


def get_profile(name=None, max_failed_drums=0):
    """Gets a training and testing profile for the RL depending on the runtype

    Args:
        name (str, optional): Type of testing profile to use. Options include: long, low_power, train, or standard_test. defaults to None. None uses a standard but unique test profile.
        max_failed_drums (int, optional): The maximum number of drums set to fail during testing (this applies a mask to randomly assign failed drums).

    Returns:
        tuple: training profile, testing_kwargs dictionary
    """
    # warn users if the profile name they provided is wrong
    if name not in ["long", "low_power", "train", "test"]:
        print(
            f"WARNING!!! {name} is not a recognized profile, defaulting to TRAIN profile."
        )

    # create interpolated power profiles
    training_profile = interp1d(
        [0, 10, 20, 30, 50, 70, 120, 140, 160, 195, 200],  # times (s)
        [1, 1, 0.98, 0.99, 0.80, 0.60, 0.60, 0.70, 0.70, 0.80, 0.80],
    )  # fraction of total power
    testing_profile = interp1d(
        [0, 10, 70, 100, 115, 125, 150, 180, 200],  # times (s)
        [1, 1, 0.50, 0.50, 0.65, 0.65, 0.50, 0.80, 0.80],
    )  # fraction of total power
    lowpower_profile = interp1d(
        [0, 5, 100, 200], [1, 1, 0.30, 0.90]  # times (s)
    )  # fraction of total power
    longtest_profile = interp1d(
        [
            0,
            2000,
            3000,
            5500,
            6000,
            10000,
            10020,
            12500,
            14000,
            16000,
            16010,
            20000,
        ],  # times (s)
        [1, 1, 0.90, 0.90, 0.45, 0.45, 0.65, 0.65, 0.80, 0.80, 0.95, 0.95],
    )  # fraction of total power

    if name == "long":
        test_profile = longtest_profile
        episode_length = (
            20000  # number of timesteps inside a single episode inside a training loop
        )
    elif name == "low_power":
        test_profile = lowpower_profile
        episode_length = 200
    elif name == "train":
        test_profile = training_profile
        episode_length = 200
    elif name == "test":
        test_profile = testing_profile
        episode_length = 200
    else:
        test_profile = testing_profile
        episode_length = 200

    training_kwargs = {
        "profile": training_profile,
        "episode_length": 200,
        "training": True,
        "max_failed_drums": 0,  # don't allow failed drums during training
        "dt": 1,
    }
    testing_kwargs = {
        "profile": test_profile,
        "episode_length": episode_length,
        "training": False,
        "max_failed_drums": max_failed_drums,  # change this later!! (not sure what they did in the original paper)
        "dt": 1,
    }

    # some error handling to make sure that episode length * dt doesn't exceed the profile's latest time

    if not helpers.episode_length_within_max_time(
        profile=training_kwargs["profile"],
        episode_length=training_kwargs["episode_length"],
        dt=training_kwargs["dt"],
    ):
        raise ValueError(f"episode_length*dt exceeds training profile's max time")

    if not helpers.episode_length_within_max_time(
        profile=testing_kwargs["profile"],
        episode_length=testing_kwargs["episode_length"],
        dt=testing_kwargs["dt"],
    ):
        raise ValueError(f"episode_length*dt exceeds testing profile's max time")

    return training_profile, testing_kwargs
