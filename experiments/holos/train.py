from pathlib import Path
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from env import Undisturbed, make_undisturbed_env
from functools import partial
import env
import shutil
import traceback
from profiles import get_profile


def train(
    model_name="default",
    env_type=env.HolosMulti,
    training_profile="train",
    episode_length=200,  # N time steps
    dt=1,  # seconds in a single time step
    total_timesteps=int(
        2e6
    ),  # the number of timesteps across parallel envs that training will explore for
    n_envs=10,  # the number of parallel environments used for training
):
    """Trains a Multi Drum RL model for the Holos Quad reactor. All drums operate independently, but a single RL controls them.

    Args:
        model_name (str, optional): Name for the folder holding this model and its associated contents. Defaults to "default".
        env_type (gym Environment, optional): A gym environment to describe the reactor. Defaults to env.HolosMulti.
        training_profile (str, optional): One of either: "long", "low_power", "train", "test" from available profiles (profiles.py). Defaults to "train".
        episode_length (int, optional): Timesteps to run a single episode for. Defaults to 200.
    """

    # check to see if we've already trained a model under this name, if so, don't train again
    model_root = Path("models")
    model_folder = model_root / model_name
    if (model_folder / "best_model.zip").exists():
        print(f"Training for {model_name} already complete!")
        return

    else:
        print(f"No existing model found for {model_name}. Beginning training...")

    # get some infrastructure set up
    model_folder.mkdir(parents=True)
    log_folder = model_folder / "logs"
    log_folder.mkdir()

    try:
        # write down the params so we have them to reference later

        with open(f"{model_folder}/params.txt", "w") as f:
            f.write(f"env type: {env_type}\n")
            f.write(f"training profile: {training_profile}\n")
            f.write(f"episode length: {episode_length}\n")
            f.write(f"dt (timestep length): {dt} seconds\n")
            f.write(f"total training timesteps (all envs): {total_timesteps}\n")
            f.write(f"number of parallel envs: {n_envs}\n")

        # set up a vectorized environment to allow for parallel training
        env_kwargs = {
            "profile": get_profile(training_profile)[0],
            "episode_length": episode_length,
            "training": True,
            "max_failed_drums": 0,  # assume drums don't fail during training
            "dt": dt,
        }
        vec_env = make_vec_env(
            partial(make_undisturbed_env, env_type),
            n_envs=n_envs,
            env_kwargs=env_kwargs,
        )
        vec_env = VecMonitor(
            vec_env, filename=None
        )  # change None if you want to save file logs, removing for now to reduce clutter

        # initialize the model
        model = sb3.PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=f"models/{model_name}/logs",
            device="cpu",
        )

        # set up the evaluation environment
        eval_env = Monitor(
            make_undisturbed_env(env_type, **env_kwargs),
            filename=f"models/{model_name}/logs",
        )

        # define the evaluation frequency per total timesteps (sum of timesteps across all parallel envs)
        eval_freq = min(
            round(1e4 / n_envs, -3), total_timesteps
        )  # rounds to the nearest 10,000th step or calls just once, dependening on size of run

        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"models/{model_name}",
            log_path=f"models/{model_name}/logs",
            deterministic=True,
            eval_freq=eval_freq,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
        return

    except Exception:
        traceback.print_exc()
        shutil.rmtree(model_folder)


if __name__ == "__main__":
    # train a model
    train(
        model_name="default",
        env_type=env.HolosMulti,
        training_profile="train",
        episode_length=200,  # N time steps
        dt=1,  # seconds in a single time step
        total_timesteps=int(
            600
        ),  # the number of timesteps across parallel envs that training will explore for
        n_envs=10,
    )
