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

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from accessories import find_latest_file, metrics
from env import Undisturbed, make_undisturbed_env
from functools import partial
from accessories import check_spec


def train_rl(env_type, save_dir, env_kwargs, total_timesteps=int(2e6), n_envs=10):
    """Train a PPO agent with periodic evaluation and checkpointing.

    Sets up `n_envs` parallel training environments and a separate
    single environment for deterministic evaluation. The actual
    training loop (rollout collection, GAE, PPO update epochs) is
    handled internally by SB3's `model.learn()`; this function only
    configures that process.

    Args:
        env_type: Environment class (not instance) to construct.
        save_dir: location to save the run's outputs
        env_kwargs: Kwargs passed to `env_type`. Must include
        total_timesteps: Total environment steps to train for,
            summed across all parallel envs.
        n_envs: Number of parallel environments for rollout collection.

    Side effects:
        Creates `save_dir/models/` and writes tensorboard logs,
        monitor CSVs, and the best checkpoint under `save_dir/logs/`
        and `save_dir/models/`.
    """
    # set up paths and directories for saving models
    model_folder = save_dir / "models/"
    model_folder.mkdir(exist_ok=True)
    log_dir = save_dir / "logs/"

    # set up a vectorized environment to allow for parallel training
    vec_env = make_vec_env(
        partial(make_undisturbed_env, env_type), 
        n_envs=n_envs, 
        env_kwargs=env_kwargs)
    vec_env = VecMonitor(vec_env, filename=str(log_dir / "vec"))

    # initialize the model
    model = sb3.PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        device="cpu",
    )

    # set up the evaluation environment
    eval_env = Monitor(make_undisturbed_env(env_type, **env_kwargs), filename=str(log_dir / "eval"))

    # define the evaluation frequency per total timesteps (sum of timesteps across all parallel envs)
    eval_freq = min(
        round(1e4 / n_envs, -3), total_timesteps
    )  # rounds to the nearest 10,000th step or calls just once, dependening on size of run

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(model_folder),
        log_path=str(log_dir),
        deterministic=True,
        eval_freq=eval_freq,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    return


def rollout(model: sb3.PPO, env, disturbance_distribution, save_histories=True):
    """Run one deterministic rollout of `model` on `env` to termination.

    Standard gym inference loop: reset, then step with the model's
    greedy action until the episode ends. `env.render()` is called
    once, after the loop — this assumes `render()` produces a
    post-hoc summary (e.g. a plot from accumulated history) rather
    than a per-frame live view. If `env`'s render mode is
    frame-based, move the render() call inside the loop.

    Args:
        model: Trained SB3 model with a `.predict()` method.
        env: Gym environment instance (not vectorized).
        disturbance_distribution (DisturbanceDistribution object from fuzzing.py): holds disturbance distributions for observations, actions, and states
        save_histories (bool, optional): decides whether to save histories from render as a csv. Defaults to True.
    """
    # get disturbance (noise) for the first observation
    xo0 = disturbance_distribution.Do.sample()
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # get disturbances for the step
        x = disturbance_distribution.sample(env.state, action)
        obs, _, terminated, truncated, _ = env.step(
            action=action,
            disturbance = x
        )
        done = terminated or truncated

    if save_histories:
        return env.render()
    else:
        return env.render(save=False)
    


def test_trained_rl(env_type, load_dir, save_dir, env_kwargs, disturbance_distribution):
    """Evaluate the most recent trained checkpoint and report control metrics.

    Loads the newest .zip checkpoint in `save_dir/models/`, runs it
    through one deterministic rollout, then reads back the resulting
    run-history CSV to compute tracking/control-effort metrics.

    Args:
        env_type: Environment class to construct for testing.
        load_dir: Path to load the model from.
        save_dir: Path to save the file to a
        env_kwargs: Kwargs passed to `env_type`
        disturbance_distribution (DisturbanceDistribution object from fuzzing.py): holds disturbance distributions for observations, actions, and states


    Returns:
        DataFrame of the full run history (state/action trace) for
        the evaluated episode.
    """
    model_folder = load_dir / "models/"
    model_path = find_latest_file(model_folder, pattern="*.zip")
    model = sb3.PPO.load(model_path, device="cpu")

    test_env = env_type(**env_kwargs, save_dir=save_dir)
    rollout(model, test_env, disturbance_distribution)

    history_path = find_latest_file(save_dir, pattern="run_history*.csv")
    history = pd.read_csv(history_path)
    mae, cae, control_effort, mean_control_effort = metrics(history)
    print(
        f"{save_dir.name} - MAE: {mae}, CAE: {cae}, "
        f"Control Effort: {control_effort}, "
        f"Mean Control Effort: {mean_control_effort}"
    )
    return history

def run_rollouts(rollout_fn, m, psi=check_spec):
    """Runs m rollouts, save the trajectories and check spec compliance.

    Args:
        rollout_fn (callable): no-arg function that runs one rollout and returns its history (pd.DataFrame)
        m (int): number of rollouts
        psi (callable): spec-check function taking a history DataFrame, returns bool

    Returns:
        histories (list[pd.DataFrame]), results (np.ndarray[bool]): trajectories and pass/fail per rollout
    """
    histories = []
    results = np.zeros(m, dtype=bool)

    for i in range(m):
        history = rollout_fn()
        histories.append(history)
        results[i] = psi(history)

    return histories, results
