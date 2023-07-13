import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt


class BeeWorld(gym.Env):
    def __init__(self, size=10, dt=1):
        self.size = size  # Room size
        self.dt = dt  # Integration timestep
        self._agent_vel = 0.0  # Translational velocity
        self._agent_theta = 0.0  # Agent's direction as angle from x-axis
        self._agent_ang_vel = 0.0  # Angular velocity

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Discrete(2),
                "smell": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
            }
        )

        # Action is a Tuple of (Translational acceleration, Angular acceleration)
        self.action_space = spaces.Tuple(
            (spaces.Box(-1, 1, dtype=float), spaces.Box(-1, 1, dtype=float))
        )

    def _check_vision(self):
        """
        Returns 1 if the bee can see the goal and 0 otherwise
        TODO: implement
        """
        return 0

    def _get_smell(self):
        """
        Returns strength of smell at agent's current location
        """
        return np.array(
            [
                np.exp(
                    -np.linalg.norm(self._agent_location - self._target_location, ord=2)
                )
            ]
        )

    def _get_obs(self):
        """
        Returns a dictionary with agent's current observations
        """
        return {"vision": self._check_vision(), "smell": self._get_smell()}

    def _get_info(self):
        """
        Provides auxiliary information
        """
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.random(size=2, dtype=float) * self.size

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = (
                self.np_random.random(size=2, dtype=float) * self.size
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        returns (observation, reward, done, info)
        """
        self._agent_location += [
            self.dt * self._agent_vel * np.sin(self._agent_theta),
            self.dt * self._agent_vel * np.cos(self._agent_theta),
        ]
        self._agent_vel += self.dt * action[0]
        self._agent_theta += self.dt * action[1]

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # TODO: Rendering needs to happen via pygame in render()
        # THIS IS TEMPORARY
        fig, ax = plt.subplots()

        ax.scatter(self._agent_location[0], self._agent_location[1], label="agent")
        ax.scatter(self._target_location[0], self._target_location[1], label="target")

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

        plt.legend()
        plt.show()

        return observation, reward, terminated, False, info


gym.register(
    id="BeeWorld",
    entry_point=BeeWorld,
    max_episode_steps=300,
)

env = gym.make("BeeWorld")
env.reset()

env.step((0.0, 0.0))
env.step((1.0, 0.0))
env.step((0.0, 0.0))
env.step((0.0, 0.0))
