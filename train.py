import numpy as np
import gymnasium as gym
from bee import BeeWorld
import torch.nn as nn

from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from gymnasium.wrappers.record_video import RecordVideo


gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=1000,
)

env = gym.make("BeeWorld", render_mode="rgb_array", max_episode_steps=1000)
env = RecordVideo(env, "video", lambda x: x % 20 == 0)
env.reset()

n_actions = 2
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

policy_kwargs = {
    "net_arch": [100, 100],  # Specify the number of hidden units per layer
    "activation_fn": nn.ReLU,  # Specify the activation function
}

model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=0.001,
    tensorboard_log="./logs/",
)
model.learn(total_timesteps=200_000, log_interval=10)
model.save("test2")
