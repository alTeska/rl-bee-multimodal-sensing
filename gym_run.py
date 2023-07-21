import numpy as np
import torch.nn as nn
from train import init_gym, init_model, setup_logging


env = init_gym("BeeWorld", logs_path="test_logs/", render_mode="human")
model = init_model(env, [100, 100], nn.ReLU, 0.01)


model.learn(total_timesteps=1000, log_interval=10)

vec_env = model.get_env()
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)

env.close()
