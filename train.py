import os
import numpy as np
import torch.nn as nn
import gymnasium as gym

from bee import BeeWorld
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=1000,
)

env = gym.make("BeeWorld", render_mode="rgb_array", max_episode_steps=1000)
env.reset()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)


models_dir = "models/{}"
model_alg = "TD3_200x200"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir.format(model_alg), exist_ok=True)

if not os.path.exists(models_dir):
    os.makedirs(logdir, exist_ok=True)

new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

## check if model esists and load it instead of creating
#  TD3.load(modelname)

model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs={
        "net_arch": [200, 200],  # Specify the number of hidden units per layer
        "activation_fn": nn.ReLU,  # Specify the activation function
    },
    learning_rate=0.01,
    tensorboard_log="./logs/",
)
model.set_logger(new_logger)
model.learn(total_timesteps=1000000, log_interval=10)

vec_env = model.get_env()
obs = vec_env.reset()

timesteps = 10000
iters = 0

while True:
    iters += 1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)

env.close()
