import os
import argparse

import numpy as np
import torch.nn as nn

import gymnasium as gym
from bee import BeeWorld

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise  # other: OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def create_directories(models_dir, model_algo, logs_dir):

    # create directories for trained model and log saving
    algo_path = models_dir + '{}'.format(model_algo)
    if not os.path.exists(algo_path):
        os.makedirs(algo_path, exist_ok=True)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)


def initialise_gym(gymName):

    # initialise Gym env
    gym.register(
        id=gymName,
        entry_point=BeeWorld,
        max_episode_steps=3000,
    )

    env = gym.make(gymName, render_mode="rgb_array")
    env.reset()

    return env

def initialise_RL_model(env, models_dir, model_algo, logs_dir, net_arch, activation_fnn, lr):


    # set up the logger and early stopping callback
    env = Monitor(env, logs_dir, allow_early_resets=True)

    logger = configure("test_logs", ["stdout", "csv", "log", "tensorboard", "json"])
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

    eval_callback = EvalCallback(env,
                                 callback_after_eval=stop_train_callback,
                                 best_model_save_path=models_dir + '{}'.format(model_algo),
                                 log_path=logs_dir,
                                 eval_freq=1000,
                                 n_eval_episodes=10,
                                 deterministic=True,
                                 render=False)


    # create the model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fnn,
        },
        learning_rate=lr
    )

    # set custom logger
    model.set_logger(logger)

    return model, eval_callback

def train(gymName='BeeWorld', base_path = 'drive/MyDrive/neuromatch/', model_algo = 'TD3', timesteps=10000, iters_max=10,
          net_arch=[100,100], activation_fnn=nn.ReLU, lr=0.01):

    print(gymName, base_path, model_algo, timesteps, iters_max, net_arch, activation_fnn)

    models_dir = base_path + 'models/'
    logs_dir = base_path + 'logs/'
    replay_buffer_dir = base_path + 'replay_buffer/'

    # initialise Gym
    env = initialise_gym(gymName)

    # initialise the RL model
    model, eval_callback = initialise_RL_model(env, models_dir, model_algo, logs_dir, net_arch, activation_fnn, lr)

    # train the RL model
    vec_env = model.get_env()
    obs = vec_env.reset()

    # training loop (+ save model at each iteration)
    iters = 0
    while iters < iters_max:
        iters += 1

        model_name = model_algo + '_' + str(timesteps*iters)
        model_path = models_dir + model_algo + '/' + model_name
        replay_buffer_path = replay_buffer_dir + model_algo + '/' + model_name

        # if we already have saved the model learning at this stage, load that model
        cur_model_zip_path = model_path + '.zip'
        if os.path.exists(cur_model_zip_path):
            print('Loading this model:', cur_model_zip_path)
            model = TD3.load(cur_model_zip_path)
            model.set_env(DummyVecEnv([lambda: gym.make("BeeWorld", render_mode="rgb_array")]))
            model.load_replay_buffer(replay_buffer_path)
        # train the model if no model saved at this stage yet
        else:
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=eval_callback)
            model.save(model_path)
            model.save_replay_buffer(replay_buffer_path)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL model.")
    parser.add_argument("--gymName", type=str, default="BeeWorld", help="Gym environment name.")
    parser.add_argument("--base_path", type=str, default="drive/MyDrive/neuromatch/", help="Base path.")
    parser.add_argument("--model_algo", type=str, default="TD3", help="RL model algorithm.")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps per iteration.")
    parser.add_argument("--iters_max", type=int, default=10, help="Maximum number of iterations.")
    parser.add_argument("--net_arch", nargs="+", type=int, default=[100, 100], help="Neural network architecture.")
    parser.add_argument("--activation_fnn", type=str, default="ReLU", help="Activation function for the neural network.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")

    args = parser.parse_args()

    # call the train function with the parsed arguments
    train(
        gymName=args.gymName,
        base_path=args.base_path,
        model_algo=args.model_algo,
        timesteps=args.timesteps,
        iters_max=args.iters_max,
        net_arch=args.net_arch,
        activation_fnn=getattr(nn, args.activation_fnn),
        lr=args.lr,
    )
