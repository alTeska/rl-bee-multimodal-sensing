import os
import yaml
import argparse
import numpy as np
import torch.nn as nn
from utils import create_directory
from model import init_gym, init_model, load_model, setup_logging
from stable_baselines3 import TD3


def new_model(
    output_path,
    gym_name,
    policy_kwargs,
    learning_rate,
):
    logs_path = os.path.join(output_path, "logs")
    video_path = os.path.join(output_path, "video")
    replay_buffer_path = os.path.join(output_path, "replay_buffer")
    create_directory(logs_path)

    env = init_gym(gym_name, logs_path=logs_path, video_path=video_path)
    callback, logger = setup_logging(env, logs_path, output_path)
    model = init_model(env, policy_kwargs, learning_rate, logger=logger)

    return model, callback


def load_existing_model(
    input_path,
    output_path,
    gym_name,
    policy_kwargs,
    learning_rate,
):
    logs_path = os.path.join(output_path, "logs")
    video_path = os.path.join(output_path, "video")

    replay_buffer_path = os.path.join(input_path, "replay_buffer")
    create_directory(output_path)
    create_directory(logs_path)

    env = init_gym(gym_name, logs_path=logs_path, video_path=video_path)
    callback, logger = setup_logging(env, logs_path, output_path)
    model = load_model(env, path, replay_buffer=replay_buffer_path, logger=logger)

    return model, callback


def custom_training(config):
    if type(config["train"]["policy_kwargs"]["activation_fn"]) == str:
        config["train"]["policy_kwargs"]["activation_fn"] = getattr(
            nn, config["train"]["policy_kwargs"]["activation_fn"]
        )
    base_path = config["setup"]["path"]
    gym_name = config["env"]["gym_name"]
    policy_kwargs = config["train"]["policy_kwargs"]
    learning_rate = config["train"]["learning_rate"]

    input_path = os.path.join(base_path, config["setup"]["old_alias"])
    output_path = os.path.join(base_path, config["setup"]["alias"])

    if config["setup"]["continue_training"] and os.path.exists(input_path):
        print("Loading existing model")

        model, callback = load_existing_model(
            input_path,
            output_path,
            gym_name,
            policy_kwargs,
            learning_rate,
        )

    else:
        print("Creating a new model")

        model, callback = new_model(
            output_path,
            gym_name,
            policy_kwargs,
            learning_rate,
        )

    model.learn(
        total_timesteps=config["train"]["timesteps"],
        reset_num_timesteps=False,
        callback=callback,
    )

    model.save_replay_buffer(os.path.join(output_path, "replay_buffer"))

    with open(os.path.join(output_path, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    env = model.get_env()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="config file for your model",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    custom_training(config)
