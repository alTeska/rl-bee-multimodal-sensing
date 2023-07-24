import os
import argparse
import yaml
import numpy as np
import torch.nn as nn
import gymnasium as gym
from bee import BeeWorld
from utils import create_directory
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FrameStack, FlattenObservation


def init_gym(
    gym_name="BeeWorld",
    render_mode="rgb_array",
    frame_stack_size=None,
    max_episode_steps=1000,
    video_path=None,
    logs_path=None,
    walls=[
        [(5.0, 0.0), (5.0, 5.0)],
    ],
    goal_size=2.0,
    agent_location_range=[(0.0, 2.0), (0.0, 10.0)],
    goal_location_range=[(5.0, 10.0), (0.0, 10.0)],
):
    """
    Initialize the Gym environment with the given setup.
    For new walls: [(pAx, pAy),(pBx,pBy), (pAx, pAy),(pBx,pBy)]

    Parameters:
        gym_name (str): The name of the custom Gym environment to initialize. Defaults to "BeeWorld".
        render_mode (str): The rendering mode for the environment. Defaults to "rgb_array".
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 1000.
        logs_path (str or None): The directory path to store logs. If None, no logging is performed. Defaults to None.
        video_path (str or None): The directory path to save video recordings. If None, no videos are recorded. Defaults to None.

    Returns:
        gym.Env: The initialized Gym environment.
    """
    gym.register(
        id=gym_name,
        entry_point=BeeWorld,
        max_episode_steps=max_episode_steps,
    )
    env = gym.make(
        gym_name,
        render_mode=render_mode,
        walls=walls,
        goal_size=goal_size,
        agent_location_range=agent_location_range,
        goal_location_range=goal_location_range,
    )

    if frame_stack_size:
        env = FlattenObservation(env)
        env = FrameStack(env, num_stack=frame_stack_size)

    if video_path:
        env = RecordVideo(env, video_path, lambda x: x % 50 == 0)

    if logs_path:
        env = Monitor(env, logs_path, allow_early_resets=True)

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
    """
    Initialize the TD3 model with the given setup.

    Parameters:
        env (gym.Env): The Gym environment to be used for training.
        policy_kwargs (dict): Dictionary containing the policy configuration. Defaults to a two-layer MLP policy.
        learning_rate (float): The learning rate for the optimizer. Defaults to 0.01.
        logger (stable_baselines3.common.logger.Logger or None): The logger to be used for logging training progress. Defaults to None.

    Returns:
        stable_baselines3.TD3: The initialized TD3 model.
    """
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",  # "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
    )

    if logger:
        model.set_logger(logger)

    return model


def load_model(
    env,
    path,
    replay_buffer=None,
    logger=None,
    learning_rate=0.001,
):
    """
    Load a pre-trained TD3 model.

    Parameters:
        env (gym.Env): The Gym environment to be used for loading the model.
        path (str): The path to the directory containing the saved model.
        replay_buffer (stable_baselines3.common.buffers.ReplayBuffer or None): The replay buffer to load into the model. Defaults to None.
        logger (stable_baselines3.common.logger.Logger or None): The logger to be used for logging training progress. Defaults to None.

    Returns:
        stable_baselines3.TD3: The loaded TD3 model.
    """
    model = TD3.load(
        os.path.join(path, "best_model"), learning_rate=lambda _: learning_rate
    )
    model.set_env(DummyVecEnv([lambda: env]))

    if replay_buffer:
        model.load_replay_buffer(replay_buffer)

    if logger:
        model.set_logger(logger)

    return model


def setup_logging(
    env,
    logs_path,
    best_model_save_path,
    max_no_improvement_evals=10,
    min_evals=5,
    eval_freq=1000,
):
    """
    Set up the logger and early stopping callback for training.

    Parameters:
        env (gym.Env): The Gym environment to be used for evaluation.
        logs_path (str): The directory path to store logs and evaluation results.
        best_model_save_path (str): The directory path to save the best model checkpoint.
        max_no_improvement_evals (int): Maximum number of evaluations without improvement before early stopping. Defaults to 10.
        min_evals (int): Minimum number of evaluations before early stopping can occur. Defaults to 5.

    Returns:
        tuple: A tuple containing the evaluation callback and logger.
            EvalCallback: The evaluation callback for the training process.
            stable_baselines3.common.logger.Logger: The logger used for logging training progress.
    """
    logger = configure(logs_path, ["stdout", "csv", "log", "tensorboard", "json"])
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=max_no_improvement_evals,
        min_evals=min_evals,
        verbose=1,
    )
    eval_callback = EvalCallback(
        env,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_save_path,
        log_path=logs_path,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    return eval_callback, logger
