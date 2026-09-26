import pytest
from pke import HolosPK
import env
import profiles
from scipy.interpolate import interp1d
import numpy as np
import viz
import loops
import fuzzing


def run_demo():
    """Runs a demo case of a training and testing loop with minimal time_steps. Expected to perform terribly!"""
    disturbance_dist = fuzzing.DisturbanceDistribution(
                Do=fuzzing.Do(sigma_p=0.05, sigma_dp=0.05, sigma_drum=0.05),
                Da=fuzzing.Da(sigma_dtheta=0),
                Ds=fuzzing.Ds(),
            )
    
    training_kwargs, testing_kwargs = profiles.get_profile("train")
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=50, n_envs=1, run_name="demo"
    )
    test_folder = run_folder / "demo"
    history = loops.test_trained_rl(
        env_type=env.HolosMulti, load_dir=run_folder, save_dir=test_folder, env_kwargs=testing_kwargs, disturbance_distribution=disturbance_dist
    )
    viz.plot_power(history, save_dir=run_folder)
    return

def run(test_profile="train", train_name="train_fivemillion", test_name=None):
    """Runs a full case of a training and testing loop.
    
    Args:
        profile (string, optional): profile you want to TEST on (all training profiles are the same). Defaults to "train"
        
    """
    training_kwargs, testing_kwargs = profiles.get_profile(test_profile)
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=int(5e6), n_envs=10, run_name=train_name
    )
    # if no test name is set up, default it to whatever the test profile is
    if test_name is None:
        test_name = test_profile + "_test"
    test_folder = run_folder / test_name
    print(test_folder)
    history = loops.test_trained_rl(
        env_type=env.HolosMulti, load_dir=run_folder, save_dir=test_folder, env_kwargs=testing_kwargs
    )
    viz.plot_power(history, save_dir=test_folder)
    return



if __name__ == "__main__":
    # run(
    #     test_profile="long",
    #     train_name="train_fivemillion",
    # )
    run_demo()
