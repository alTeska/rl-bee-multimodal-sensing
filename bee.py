import numpy as np
import pygame
import gymnasium as gym
from pygame.locals import *
from gymnasium import spaces


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

        pygame.init()
        self.screen_size = (400, 400)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("BeeWorld")

        self.clock = pygame.time.Clock()

        self.trajectory = []

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

        self.trajectory = []  # Reset trajectory

        return observation, info

    def step(self, action):
        """
        returns (observation, reward, done, info)
        """
        # if not self.action_space.contains(action):
        # raise ValueError(f'Invalid action {action} ({type(action)})')

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

        self.trajectory.append(self._agent_location.copy())

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()

        self.render()
        self.clock.tick(60)

        return observation, reward, terminated, False, info

    def render(self):
        """
        Renders the current state of the environment using Pygame
        """
        self.screen.fill((255, 255, 255))

        agent_pos = self._agent_location * (self.screen_size[0] / self.size)
        target_pos = self._target_location * (self.screen_size[0] / self.size)

        pygame.draw.circle(self.screen, (255, 0, 0), agent_pos.astype(int), 5)
        pygame.draw.circle(self.screen, (0, 255, 0), target_pos.astype(int), 5)

        if len(self.trajectory) > 1:
            trajectory_points = [
                pos * (self.screen_size[0] / self.size) for pos in self.trajectory
            ]
            pygame.draw.lines(self.screen, (0, 0, 255), False, trajectory_points, 2)

        pygame.display.flip()

    def close(self):
        """
        Clean up the environment
        """
        pygame.quit()
