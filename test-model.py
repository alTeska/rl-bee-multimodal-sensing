import os
import yaml
import argparse
import gymnasium as gym
from tqdm.notebook import trange
from stable_baselines3 import TD3
from model import init_gym, load_model
from render_model import render_prediction

config_path = "configs/test-config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

output_path = os.path.join(config["setup"]["path"], config["setup"]["alias"])

# Load the model and generate prediction frames
env = init_gym(
    gym_name=config["env"]["gym_name"],
    render_mode=config["env"]["render_mode"],
    video_path=os.path.join(output_path, "video"),
    logs_path=None,
    walls=config["env"]["walls"],
    goal_size=config["env"]["goal_size"],
    agent_location_range=config["env"]["agent_location_range"],
    goal_location_range=config["env"]["goal_location_range"],
    frame_stack_size=config["env"]["frame_stack_size"],
    noise_smell=config["env"]["noise_smell"],
    noise_vision=config["env"]["noise_vision"],
)

model = load_model(env, output_path, replay_buffer=None, logger=None)
frames = render_prediction(model, config["test"]["prediction_steps"])
