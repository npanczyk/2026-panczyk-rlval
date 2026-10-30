import numpy as np
from pathlib import Path
import stable_baselines3 as sb3
from veery.accessories import check_spec
import env
import profiles
import numpy as np
from veery import viz
from veery import fuzzing
from tqdm import tqdm
import helpers


def rollout(
    model: sb3.PPO,
    env_type,
    env_kwargs,
    disturbance_distribution,
    results_folder,
    save_histories=False,
):
    """Run one deterministic rollout of `model` on `env` to termination.

    Standard gym inference loop: reset, then step with the model's
    greedy action until the episode ends. `env.render()` is called
    once, after the loop — this assumes `render()` produces a
    post-hoc summary (e.g. a plot from accumulated history) rather
    than a per-frame live view.

    Args:
        model: Trained SB3 model with a `.predict()` method.
        env: Gym environment instance (not vectorized).
        disturbance_distribution (DisturbanceDistribution object from fuzzing.py): holds disturbance distributions for observations, actions, and states
        results_folder (str): where to save the histories and aux info
        save_histories (bool, optional): decides whether to save histories from render as a csv. Defaults to True.
    """
    test_env = env_type(**env_kwargs, save_dir=results_folder)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # get disturbances for the step
        x = disturbance_distribution.sample(test_env.state, action)
        obs, _, terminated, truncated, _ = test_env.step(action=action, disturbance=x)
        done = terminated or truncated

    if save_histories:
        return test_env.render(save=True)
    else:
        return test_env.render(save=False)


#     if save_histories:
#         mae, cae, control_effort, mean_control_effort = metrics(history)
#         print(
#             f"{save_dir.name} - MAE: {mae}, CAE: {cae}, "
#             f"Control Effort: {control_effort}, "
#             f"Mean Control Effort: {mean_control_effort}"
#         )
#     return history


def run_rollouts(test_profile, model_name, run_name, m, p_sig=0.01, psi=check_spec):
    """Runs m rollouts, save the trajectories and check spec compliance.

    Args:
        test_profile (): from profiles
        model_name (str): model name inside /models folder
        run_name (str): name of folder inside /results to save the results
        m (int): number of rollouts
        psi (callable): spec-check function taking a history DataFrame, returns bool

    Returns:
        histories (list[pd.DataFrame]), results (np.ndarray[bool]): trajectories and pass/fail per rollout
    """
    _, testing_kwargs = profiles.get_profile(test_profile)
    results_root = Path("results")
    results_folder = results_root / run_name
    results_folder.mkdir(parents=True, exist_ok=True)
    plots_folder = results_folder / "figures"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # set up disturbance distribution for the test
    disturbance_dist = fuzzing.DisturbanceDistribution(
        Do=fuzzing.Do(sigma_p=p_sig, sigma_drum=0.005),
        Da=fuzzing.Da(sigma_dtheta=0),
        Ds=fuzzing.Ds(),
    )

    histories = []
    results = np.zeros(m, dtype=bool)
    model = helpers.load_trained_model(model_name)

    for i in tqdm(range(m)):
        history = rollout(
            model=model,
            env_type=env.HolosMulti,
            env_kwargs=testing_kwargs,
            disturbance_distribution=disturbance_dist,
            results_folder=results_folder,
            save_histories=False,
        )
        histories.append(history)
        results[i] = psi(history)

    viz.plot_rollouts(histories, results, save_dir=plots_folder)
    viz.plot_robustness(
        history, threshold=0.03, time=None, ax=None, save_dir=plots_folder
    )

    return histories, results


# def rollout_from_snapshot(env, model, snapshot, disturbance_distribution):
#     """Runs a rollout of the model from a snapshot for MCTS.

#     Args:
#         env (holos multi environment): _description_
#         model (sb3.PPO model object): loaded pre-trained model used on the entire rollout
#         snapshot (dict): snapshot of the environment state
#         disturbance_distribution (Disturbance object from fuzzing.py)
#     Returns:
#         tuple: actual power over rollout, desired power over rollout
#     """
#     observation = env.set_snapshot(snapshot)
#     p_actual_list = []
#     p_desired_list = []
#     done = False
#     while env.time < env.runtime and not done:
#         action, _ = model.predict(observation, deterministic=True)
#         x = disturbance_distribution.sample(env.state, action)
#         observation, _, terminated, truncated, _ = env.step(action, disturbance=x)
#         p_desired_list.append(float(env.profile(env.time)))
#         p_actual_list.append(env._p)
#         done = terminated or truncated
#     return p_actual_list, p_desired_list


def overlapping_rollouts(test_profile, model_name, run_name):
    """_summary_

    Args:
        test_profile (_type_): _description_
        model_name (_type_): _description_
        run_name (_type_): _description_
    """
    histories_list = []
    results_list = []
    sigmas = [0.01, 0.005, 0.0025, 0.00125]
    colors = ["cyan", "magenta", "blue", "tomato"]
    for sig in sigmas:
        h, r = run_rollouts(
            test_profile=test_profile,
            model_name=model_name,
            run_name=run_name,
            m=5,
            p_sig=sig,
        )
        histories_list.append(h)
        results_list.append(r)

    viz.plot_overlapping_rollouts(
        histories_list,
        results_list,
        sigmas,
        colors,
        threshold=0.03,
        save_as=f"results/{run_name}/figures/overlapping.png",
        output_name="Fraction of Total Power, [-]",
    )
    return


if __name__ == "__main__":
    run_rollouts(
        test_profile="test", model_name="default", run_name="demo", m=1, p_sig=0.01
    )
    overlapping_rollouts(test_profile="test", model_name="default", run_name="demo")
