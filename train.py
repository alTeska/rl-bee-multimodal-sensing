import os
import yaml
import argparse
import numpy as np
import torch.nn as nn
from utils import create_directory, save_config
from model import init_gym, init_model, load_model, setup_logging
from stable_baselines3 import TD3


def custom_training(config):
    if type(config["train"]["policy_kwargs"]["activation_fn"]) == str:
        config["train"]["policy_kwargs"]["activation_fn"] = getattr(
            nn, config["train"]["policy_kwargs"]["activation_fn"]
        )

    base_path = config["setup"]["path"]
    input_path = os.path.join(base_path, config["setup"]["old_alias"])
    output_path = os.path.join(base_path, config["setup"]["alias"])

    logs_path = os.path.join(output_path, "logs")
    video_path = os.path.join(output_path, "video")
    create_directory(logs_path)
    create_directory(video_path)

    env = init_gym(
        gym_name=config["env"]["gym_name"],
        logs_path=logs_path,
        video_path=None,
        render_mode=config["env"]["render_mode"],
    )

    callback, logger = setup_logging(env, logs_path, output_path)

    if config["setup"]["continue_training"] and os.path.exists(input_path):
        print("Loading existing model")

        replay_buffer_path = os.path.join(input_path, "replay_buffer")
        model = load_model(
            env, input_path, replay_buffer=replay_buffer_path, logger=logger
        )

    else:
        print("Creating a new model")

        model = init_model(
            env=env,
            policy_kwargs=config["train"]["policy_kwargs"],
            learning_rate=config["train"]["learning_rate"],
            logger=logger,
        )

    model.learn(
        total_timesteps=config["train"]["timesteps"],
        reset_num_timesteps=False,
        callback=callback,
    )

    # Save replay bugger and config
    model.save_replay_buffer(os.path.join(output_path, "replay_buffer"))
    save_config(config, output_path)

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
