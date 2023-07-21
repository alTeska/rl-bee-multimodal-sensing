import os
import yaml
import argparse
import gymnasium as gym
from tqdm.notebook import trange
from stable_baselines3 import TD3
from model import init_gym, load_model


def render_prediction(
    model,
    prediction_steps=1000,
):
    """Load the existing model and generate the prediction frames"""
    vec_env = model.get_env()
    obs = vec_env.reset()

    frames = []
    for i in trange(prediction_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        frames.append(vec_env.render())

    return frames


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

    output_path = os.path.join(config["setup"]["path"], config["setup"]["alias"])

    # load model
    env = init_gym(
        gym_name=config["env"]["gym_name"],
        render_mode=config["env"]["render_mode"],
        video_path=os.path.join(output_path, "video"),
        logs_path=None,
    )

    model = load_model(env, output_path, replay_buffer=None, logger=None)
    frames = render_prediction(model, config["test"]["prediction_steps"])
