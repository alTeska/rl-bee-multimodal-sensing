import numpy as np
import torch.nn as nn
from model import init_gym, init_model


env = init_gym(
    "BeeWorld",
    logs_path="test_logs/",
    render_mode="human",
    walls=[
        [[5.0, 0.0], [5.0, 4.0]],
        [[2.5, 10.0], [2.5, 6.0]],
        [[7.5, 10.0], [7.5, 6.0]],
    ],
    goal_size=0.5,
    agent_location_range=[[0.0, 2.0], [0.0, 10.0]],
    goal_location_range=[[5.0, 10.0], [0.0, 10.0]],
    frame_stack_size=5,
    noise_vision=True,
    noise_smell=True,
)


model = init_model(
    env, policy_kwargs={"net_arch": [100, 100], "activation_fn": nn.ReLU}
)

model.learn(total_timesteps=1000, log_interval=10)

vec_env = model.get_env()
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(obs.shape)

env.close()
