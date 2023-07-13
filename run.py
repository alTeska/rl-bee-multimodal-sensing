import numpy as np
import gymnasium as gym
from bee import BeeWorld


gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=3000,
)

env = gym.make("BeeWorld", render_mode="human")
env.reset()

done = False


# env.step((0.0, 0.0))
while not done:
    A = np.random.uniform(-0.5, 0.5)
    theta = np.random.uniform(-0.5, 0.5)
    env.step((A, theta))

env.close()
