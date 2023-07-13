import numpy as np
import gymnasium as gym
from bee import BeeWorld


gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=300,
)

env = gym.make("BeeWorld")
env.reset()

done = False


env.step((0.0, 0.0))
while not done:
    A = np.random.uniform(-2, 2)
    theta = np.random.uniform(-2, 2)
    env.step((A, theta))

env.close()
