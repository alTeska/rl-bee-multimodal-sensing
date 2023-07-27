import os
import yaml
import argparse
import gymnasium as gym
from tqdm.notebook import trange
from stable_baselines3 import TD3
from model import init_gym, load_model
from render_model import render_prediction
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Path to the configuration file
config_path = "configs/test-config.yaml"

# Load configuration parameters from the YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Set the output path for test results and logs
output_path = os.path.join(config["setup"]["path"], config["setup"]["alias"])
log_path = os.path.join(output_path, "test_logs")
os.makedirs(log_path, exist_ok=True)

# Initialize the Gym environment for testing the model's predictions
env = init_gym(
    gym_name=config["env"]["gym_name"],
    render_mode=config["env"]["render_mode"],
    video_path=None,
    logs_path=os.path.join(output_path, "test_logs"),
    walls=config["env"]["walls"],
    goal_size=config["env"]["goal_size"],
    agent_location_range=config["env"]["agent_location_range"],
    goal_location_range=config["env"]["goal_location_range"],
    frame_stack_size=config["env"]["frame_stack_size"],
    noise_smell=config["env"]["noise_smell"],
    noise_vision=config["env"]["noise_vision"],
    max_episode_steps=config["test"]["max_episode_steps"],
)

# Load the pre-trained model for evaluation
model = load_model(env, output_path, replay_buffer=None, logger=None)

# Evaluate the model's performance
res = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=config["test"]["eval_episodes"],
    deterministic=True,
    render=False,
    return_episode_rewards=True,
)

# Save the evaluation results to a log file
log = os.path.join(log_path, f"{config['test']['log_name']}.txt")
np.savetxt(log, np.array(res))
