import os
import yaml
import argparse
import numpy as np
import torch.nn as nn
from utils import create_directory, save_config
from model import init_gym, init_model, load_model, setup_logging
from stable_baselines3 import TD3


def custom_training(config):
    """
    Train a custom reinforcement learning model using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    Parameters:
        config (dict): A dictionary containing configuration parameters for training the model.

    Config Dictionary Structure:
        {
            "train": {
                "policy_kwargs": {
                    "activation_fn": str or nn.Module,
                    ... other policy_kwargs ...
                },
                "learning_rate": float,
                "timesteps": int
            },
            "setup": {
                "path": str,
                "old_alias": str,
                "alias": str,
                "continue_training": bool
            },
            "env": {
                "gym_name": str,
                "render_mode": str
            }
        }

    The function performs the following steps:
    1. Creates the necessary directories for logs and videos.
    2. Initializes the Gym environment for training.
    3. Loads an existing model if `config["setup"]["continue_training"]` is True and the model exists.
       Otherwise, creates a new model.
    4. Trains the model for the specified number of timesteps.
    5. Saves the replay buffer and configuration.

    Args:
        config (dict): A dictionary containing configuration parameters for training the model.

    Returns:
        None
    """
    if type(config["train"]["policy_kwargs"]["activation_fn"]) == str:
        # Convert the activation function name to the corresponding nn.Module class
        config["train"]["policy_kwargs"]["activation_fn"] = getattr(
            nn, config["train"]["policy_kwargs"]["activation_fn"]
        )

    # Set up paths for input and output directories
    base_path = config["setup"]["path"]
    input_path = os.path.join(base_path, config["setup"]["old_alias"])
    output_path = os.path.join(base_path, config["setup"]["alias"])

    # Create directories for logs and videos
    logs_path = os.path.join(output_path, "logs")
    create_directory(logs_path)

    video_path = None
    if config["env"]["video"]:
        video_path = os.path.join(output_path, "video")
        create_directory(video_path)

    # Initialize the Gym environment for training
    env = init_gym(
        gym_name=config["env"]["gym_name"],
        logs_path=logs_path,
        video_path=video_path,
        render_mode=config["env"]["render_mode"],
        max_episode_steps=config["train"]["max_episode_steps"],
    )

    env_eval = init_gym(
        gym_name="EvaluationGym",
        logs_path=logs_path,
        video_path=None,
        render_mode=config["env"]["render_mode"],
        max_episode_steps=config["train"]["max_episode_steps"],
    )
    # Set up logging for training progress
    callback, logger = setup_logging(
        env_eval,
        logs_path,
        output_path,
        max_no_improvement_evals=config["train"]["max_no_improvement_evals"],
    )

    if config["setup"]["continue_training"] and os.path.exists(input_path):
        print("Loading existing model")
        # Load the existing model for further training
        replay_buffer_path = os.path.join(input_path, "replay_buffer")
        model = load_model(
            env, input_path, replay_buffer=replay_buffer_path, logger=logger
        )
    else:
        print("Creating a new model")
        # Create a new model for training
        model = init_model(
            env=env,
            policy_kwargs=config["train"]["policy_kwargs"],
            learning_rate=config["train"]["learning_rate"],
            logger=logger,
        )

    # Train the model
    model.learn(
        total_timesteps=config["train"]["timesteps"],
        reset_num_timesteps=False,
        callback=callback,
    )

    # Save the replay buffer and configuration
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
