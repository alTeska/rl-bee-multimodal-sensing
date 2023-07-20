import os
import argparse
from tqdm.notebook import trange

import gymnasium as gym

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv


def retrieve_last_model_saved(base_path="drive/MyDrive/neuromatch/", model_algo="TD3"):
    models_dir = base_path + "models/"
    saved_models_path = models_dir + model_algo

    max_number = float("-inf")
    max_filename = ""

    # Loop over the files in the folder
    for filename in os.listdir(saved_models_path):
        if filename.endswith(".zip"):
            number = int(filename.split(".")[0])

            if number > max_number:
                max_number = number
                max_filename = filename

    return max_filename


def test(
    gym_name="BeeWorld",
    base_path="drive/MyDrive/neuromatch/",
    model_algo="TD3",
    range_max=1000,
):
    # load the latest model saved during training
    latest_model_path = retrieve_last_model_saved(base_path, model_algo)

    loaded_model = TD3.load(latest_model_path)
    loaded_model.set_env(
        DummyVecEnv([lambda: gym.make(gym_name, render_mode="rgb_array")])
    )

    vec_env = loaded_model.get_env()
    obs = vec_env.reset()

    frames = []
    for i in trange(range_max):
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        frames.append(vec_env.render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL model.")
    parser.add_argument(
        "--gym_name", type=str, default="BeeWorld", help="Gym environment name."
    )
    parser.add_argument(
        "--base_path", type=str, default="drive/MyDrive/neuromatch/", help="Base path."
    )
    parser.add_argument(
        "--model_algo", type=str, default="TD3", help="RL model algorithm."
    )
    parser.add_argument(
        "--range_max",
        type=int,
        default=1000,
        help="Max range to use during evaluation.",
    )

    args = parser.parse_args()

    # call the test function with the parsed arguments
    test(
        gym_name=args.gym_name,
        base_path=args.base_path,
        model_algo=args.model_algo,
        range_max=args.range_max,
    )
