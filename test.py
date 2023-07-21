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
    """
    Generate prediction frames using a trained RL model.

    This function uses the given RL model to predict actions for a given number of steps and
    generates frames for each step by rendering the environment. It returns a list of frames.

    Parameters:
        model (stable_baselines3.TD3): The trained RL model.
        prediction_steps (int, optional): The number of steps to generate predictions and frames.
                                          Defaults to 1000.

    Returns:
        list: A list of rendered frames from the environment during prediction.
    """
    vec_env = model.get_env()
    obs = vec_env.reset()

    frames = []
    for i in trange(prediction_steps):
        # Predict the action based on the observation; Perform the action in the environment
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        # Render the environment and add the frame to the frames list
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

    # Load the model and generate prediction frames
    env = init_gym(
        gym_name=config["env"]["gym_name"],
        render_mode=config["env"]["render_mode"],
        video_path=os.path.join(output_path, "video"),
        logs_path=None,
    )

    model = load_model(env, output_path, replay_buffer=None, logger=None)
    frames = render_prediction(model, config["test"]["prediction_steps"])
