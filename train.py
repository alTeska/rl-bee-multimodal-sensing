import os
import argparse
import numpy as np
import torch.nn as nn
import gymnasium as gym
from bee import BeeWorld
from utils import create_directory
from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    # OrnsteinUhlenbeckActionNoise
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def init_gym(gym_name, logs_path=None, render_mode="rgb_array"):
    """Initialise the gym environment with given setup"""
    gym.register(
        id=gym_name,
        entry_point=BeeWorld,
        max_episode_steps=3000,
    )

    env = gym.make(gym_name, render_mode=render_mode)
    env.reset()

    if logs_path:
        env = Monitor(env, logs_path, allow_early_resets=True)

    return env


def init_model(
    env,
    policy_kwargs={
        "net_arch": [100, 100],
        "activation_fn": nn.ReLU,
    },
    learning_rate=0.01,
    logger=None,
):
    """Initialise the model with given setup"""
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs={
            "net_arch": net_arch,
            "activation_fn": activation_fnn,
        },
        learning_rate=learning_rate,
    )

    if logger:
        model.set_logger(logger)

    return model


def setup_logging(env, logs_path, best_model_save_path):
    logger = configure("test_logs", ["stdout", "csv", "log", "tensorboard", "json"])
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3, min_evals=5, verbose=1
    )
    """Set up the logger and early stopping callback"""
    eval_callback = EvalCallback(
        env,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_save_path,
        log_path=logs_path,
        eval_freq=1000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    return eval_callback, logger


def train(
    gym_name="BeeWorld",
    base_path="drive/MyDrive/neuromatch/",
    model_algo="TD3",
    timesteps=10000,
    iters_max=10,
    net_arch=[100, 100],
    activation_fnn=nn.ReLU,
    learning_rate=0.01,
):
    # TODO: maybe just expose the policy_kwargs as a parameter?

    print(
        gym_name, base_path, model_algo, timesteps, iters_max, net_arch, activation_fnn
    )

    models_path = base_path + "models/"
    logs_path = base_path + "logs/"
    replay_buffer_path = base_path + "replay_buffer/"
    best_model_save_path = models_path + "{}".format(model_algo)

    create_directory(models_path)
    create_directory(logs_path)
    create_directory(replay_buffer_path)
    create_directory(best_model_save_path)

    # initialise Gym
    env = init_gym(gym_name, logs_path)
    callback, logger = setup_logging(env, logs_path, best_model_save_path)

    policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fnn}
    model = init_model(env, policy_kwargs, learning_rate, logger=logger)

    # train the RL model
    vec_env = model.get_env()
    obs = vec_env.reset()

    # training loop (+ save model at each iteration)
    iters = 0
    while iters < iters_max:
        iters += 1

        model_name = model_algo + "_" + str(timesteps * iters)
        model_path = models_path + model_algo + "/" + model_name
        replay_buffer_path = replay_buffer_path + model_algo + "/" + model_name

        cur_model_zip_path = model_path + ".zip"

        # if we already have saved the model learning at this stage, load that model
        # TODO: it is a bit akward, cause we just retrained the model and then check if exists and pick the old model?
        if os.path.exists(cur_model_zip_path):
            print("Loading this model:", cur_model_zip_path)
            model = TD3.load(cur_model_zip_path)
            model.set_env(
                DummyVecEnv([lambda: gym.make("BeeWorld", render_mode="rgb_array")])
            )
            model.load_replay_buffer(replay_buffer_path)

        # train the model if no model saved at this stage yet
        else:
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                callback=callback,
            )
            model.save(model_path)
            model.save_replay_buffer(replay_buffer_path)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL model.")
    ## TODO: do we really care about the gym name
    parser.add_argument(
        "--gym_name", type=str, default="BeeWorld", help="Gym environment name."
    )
    parser.add_argument(
        "--base_path", type=str, default="drive/MyDrive/neuromatch/", help="Base path."
    )
    ## TODO: do we need to expose the model algorithm like this?
    parser.add_argument(
        "--model_algo", type=str, default="TD3", help="RL model algorithm."
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Total timesteps per iteration."
    )
    parser.add_argument(
        "--iters_max", type=int, default=10, help="Maximum number of iterations."
    )
    parser.add_argument(
        "--net_arch",
        nargs="+",
        type=int,
        default=[100, 100],
        help="Neural network architecture.",
    )
    parser.add_argument(
        "--activation_fnn",
        type=str,
        default="ReLU",
        help="Activation function for the neural network.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")

    args = parser.parse_args()

    # call the train function with the parsed arguments
    train(
        gym_name=args.gym_name,
        base_path=args.base_path,
        model_algo=args.model_algo,
        timesteps=args.timesteps,
        iters_max=args.iters_max,
        net_arch=args.net_arch,
        activation_fnn=getattr(nn, args.activation_fnn),
        learning_rate=args.lr,
    )
