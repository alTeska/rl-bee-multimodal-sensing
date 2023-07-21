import os
import argparse
import yaml
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
from gymnasium.wrappers.record_video import RecordVideo


def init_gym(
    gym_name="BeeWorld",
    render_mode="rgb_array",
    max_episode_steps=1000,
    logs_path=None,
    video_path=None,
):
    """Initialise the gym environment with given setup"""
    gym.register(
        id=gym_name,
        entry_point=BeeWorld,
        max_episode_steps=max_episode_steps,
    )
    env = gym.make(gym_name, render_mode=render_mode)

    if logs_path:
        env = Monitor(env, logs_path, allow_early_resets=True)

    if video_path:
        env = RecordVideo(env, video_path, lambda x: x % 10 == 0)

    env.reset()

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
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
    )

    if logger:
        model.set_logger(logger)

    return model


def load_model(env, path, replay_buffer=None, logger=None):
    # load model
    model = TD3.load(os.path.join(path, "best_model"))
    model.set_env(DummyVecEnv([lambda: env]))

    if replay_buffer:
        model.load_replay_buffer(replay_buffer)

    if logger:
        model.set_logger(logger)

    return model


def setup_logging(
    env, logs_path, best_model_save_path, max_no_improvement_evals=10, min_evals=5
):
    logger = configure(logs_path, ["stdout", "csv", "log", "tensorboard", "json"])
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=max_no_improvement_evals,
        min_evals=min_evals,
        verbose=1,
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
