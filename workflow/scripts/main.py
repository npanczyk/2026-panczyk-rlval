import pytest
from pke import HolosPK
import env
import profiles
from scipy.interpolate import interp1d
import numpy as np
import viz
import loops
import fuzzing
from accessories import check_spec


def run_demo():
    """Runs a demo case of a training and testing loop with minimal time_steps. Expected to perform terribly!"""
    disturbance_dist = fuzzing.DisturbanceDistribution(
                Do=fuzzing.Do(sigma_p=0.05, sigma_drum=0.05),
                Da=fuzzing.Da(sigma_dtheta=0),
                Ds=fuzzing.Ds(),
            )
    
    training_kwargs, testing_kwargs = profiles.get_profile("train")
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=50, n_envs=1, run_name="demo"
    )
    test_folder = run_folder / "demo"
    history = loops.test_trained_rl(
        env_type=env.HolosMulti, load_dir=run_folder, save_dir=test_folder, env_kwargs=testing_kwargs, disturbance_distribution=disturbance_dist, save_histories=True
    )
    viz.plot_power(history, save_dir=run_folder)
    return

def run_rollouts(test_profile, train_name, m, psi=check_spec):
    """Runs m rollouts, save the trajectories and check spec compliance.

    Args:
        rollout_fn (callable): no-arg function that runs one rollout and returns its history (pd.DataFrame)
        m (int): number of rollouts
        psi (callable): spec-check function taking a history DataFrame, returns bool

    Returns:
        histories (list[pd.DataFrame]), results (np.ndarray[bool]): trajectories and pass/fail per rollout
    """
    training_kwargs, testing_kwargs = profiles.get_profile(test_profile)
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=int(5e6), n_envs=10, run_name=train_name
    )
    # if no test name is set up, default it to whatever the test profile is
    test_name = test_profile + "_fuzzed_test"
    test_folder = run_folder / test_name

    # set up disturbance distribution for the test
    disturbance_dist = fuzzing.DisturbanceDistribution(
                        Do=fuzzing.Do(sigma_p=0.01, sigma_drum=0.005),
                        Da=fuzzing.Da(sigma_dtheta=0),
                        Ds=fuzzing.Ds(),
                    )
    
    histories = []
    results = np.zeros(m, dtype=bool)

    for i in range(m):
        print(f"Round {i}")
        history = loops.test_trained_rl(
        env_type=env.HolosMulti, load_dir=run_folder, save_dir=test_folder, env_kwargs=testing_kwargs, disturbance_distribution=disturbance_dist, save_histories=False
    )
        histories.append(history)
        results[i] = psi(history)

    viz.plot_rollouts(histories, results, save_dir=test_folder)

    return histories, results


if __name__ == "__main__":
    run_rollouts(
        test_profile="test",
        train_name="train_fivemillion",
        m=5,
    )
    # run_demo()
