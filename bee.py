import numpy as np
import pygame
import gymnasium as gym
from pygame.locals import *
from gymnasium import spaces


def cone_locations(agent_location, agent_theta, cone_phi, scale_factor):
    """Calculate the positions of two points forming a  vision cone around the agent for the visualization

    Args:
        agent_location (numpy.ndarray): A 1D numpy array containing the (x, y) coordinates of the agent.
        agent_theta (float): The angle (in radians) representing the agent's orientation.
        cone_phi (float): The half-angle (in radians) of the cone from the agent's direction.
        scale_factor (float): A scaling factor applied to the calculated points.

    Returns:
        tuple: A tuple containing two 1D numpy arrays, 'point_left' and 'point_right', which represent the
                positions of the left and right points forming the cone, respectively.
    """
    point_left = (
        agent_location
        + [
            2 * np.cos(agent_theta + cone_phi),
            2 * np.sin(agent_theta + cone_phi),
        ]
    ) * scale_factor

    point_right = (
        agent_location
        + [
            2 * np.cos(agent_theta - cone_phi),
            2 * np.sin(agent_theta - cone_phi),
        ]
    ) * scale_factor

    return point_left, point_right


class BeeWorld(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, size=10, dt=0.1, render_mode="human", max_episode_steps=1000):
        self.dtype = "float32"
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.steps = 0

        self.size = size  # Room size
        self.dt = dt  # Integration timestep
        self._agent_vel = 0.0  # Translational velocity
        self._agent_theta = 0.0  # Agent's direction as angle from x-axis
        self._agent_ang_vel = 0.0  # Angular velocity

        self.linear_acceleration_range = 0.5
        self.angular_acceleration_range = 0.1

        self.cone_phi = np.pi / 8  # Vision cone angle

        self.walls = [
            (0, 0),
            (self.size, 0),
            (0, self.size),
            (self.size, self.size),
        ]

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Discrete(2),
                "smell": spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                "velocity": spaces.Box(-1, 1, shape=(2,), dtype=self.dtype),
                "time": spaces.Box(
                    low=0,
                    high=self.max_episode_steps,
                    dtype=self.dtype,
                ),
            }
        )

        # Action is a Tuple of (Translational acceleration, Angular acceleration)
        self.action_space = spaces.Box(
            -1,
            1,
            dtype=self.dtype,
            shape=(2,),
        )

        self.screen: pygame.Surface = None
        self.clock = None
        self.screen_size = (400, 400)

        self.trajectory = []

    def _check_vision(self):
        """
        Returns 1 if the bee can see the goal and 0 otherwise
        TODO: add walls
        """
        ray = self._target_location - self._agent_location  # raycast from agent to goal
        ang = (np.arctan2(ray[0], ray[1])) % (2 * np.pi)  # angle of raycast

        diff = np.abs(ang - self._agent_theta)

        if (diff < self.cone_phi) or ((2 * np.pi - diff) < self.cone_phi):
            return 1

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
        return {
            "vision": self._check_vision(),
            "smell": self._get_smell(),
            "velocity": np.array(
                [self._agent_vel, self._agent_ang_vel], dtype=self.dtype
            ),
            "time": self._get_time(),
        }

    def _get_time(self):
        self.steps += 1
        return self.steps / self.max_episode_steps

    def _get_info(self):
        """
        Provides auxiliary information
        """
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_vel = 0.0  # Translational velocity
        self._agent_theta = 0.0  # Agent's direction as angle from x-axis
        self._agent_ang_vel = 0.0  # Angular velocity

        self._agent_location = (
            self.np_random.random(size=2, dtype=self.dtype) * self.size
        )

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.random(size=2, dtype=self.dtype)

        # location close to the target
        # self._target_location = self._agent_location + (
        # self.np_random.random(size=2, dtype=self.dtype) * self.size * 0.3
        # )

        observation = self._get_obs()
        info = self._get_info()

        self.trajectory = []  # Reset trajectory

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """
        Returns (observation, reward, done, info)
        """

        old_agent_location = self._agent_location.copy()

        self._agent_location += [
            self.dt * self._agent_vel * np.cos(self._agent_theta),
            self.dt * self._agent_vel * np.sin(self._agent_theta),
        ]
        self._agent_vel += self.dt * action[0] * self.linear_acceleration_range
        self._agent_theta += self.dt * self._agent_ang_vel
        self._agent_ang_vel += self.dt * action[1] * self.angular_acceleration_range

        self._agent_ang_vel = np.clip(self._agent_ang_vel, -1, 1)
        self._agent_vel = np.clip(self._agent_vel, -1, 1)

        self._agent_theta = self._agent_theta % (2 * np.pi)

        # Check if the agent is outside the valid range
        if any(self._agent_location < 0) or any(self._agent_location > self.size):
            self._agent_location = old_agent_location

        goal_distance = np.linalg.norm(
            self._target_location - self._agent_location, ord=2
        )

        terminated = goal_distance < 1

        observation = self._get_obs()

        # Rewards
        reward = 100 if terminated else 0  # Binary sparse rewards
        reward += observation["smell"][0]
        reward += observation["vision"]
        reward -= 0.1 * np.sum(np.abs(action) ** 2)  # Energy expenditure
        reward -= observation["time"]  # time passed

        print(reward)

        info = self._get_info()

        self.trajectory.append(self._agent_location.copy())

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    quit()

            self.render()

        return observation, reward, terminated, False, info

    def render(self, scale=0.9):
        """
        Renders the current state of the environment using Pygame.
        The screen is scaled Calculate the 90% screen size, the positions are also transformed based on the scale factor.
        """
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("BeeWorld")

        self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(self.screen_size)

        self.surf.fill((255, 255, 255))

        screen_width = int(self.screen_size[0] * scale)
        screen_height = int(self.screen_size[1] * scale)
        screen_offset_x = int((self.screen_size[0] - screen_width) / 2)
        screen_offset_y = int((self.screen_size[1] - screen_height) / 2)

        scale_factor = screen_width / self.size

        agent_pos = self._agent_location * scale_factor
        agent_pos += np.array([screen_offset_x, screen_offset_y])
        target_pos = self._target_location * scale_factor
        target_pos += np.array([screen_offset_x, screen_offset_y])

        pygame.draw.circle(self.surf, (255, 0, 0), agent_pos.astype(int), 5)
        pygame.draw.circle(
            self.surf, (0, 255, 0), target_pos.astype(int), 1 * scale_factor
        )

        if len(self.trajectory) > 1:
            trajectory_points = [
                pos * scale_factor + np.array([screen_offset_x, screen_offset_y])
                for pos in self.trajectory
            ]
            pygame.draw.lines(self.surf, (0, 0, 255), False, trajectory_points, 2)

        pointl, pointr = cone_locations(
            self._agent_location, self._agent_theta, self.cone_phi, scale_factor
        ) + np.array([screen_offset_x, screen_offset_y])

        pygame.draw.lines(
            self.surf,
            (255, 0, 0),
            False,
            [
                agent_pos.astype(int),
                pointl.astype(int),
            ],
            2,
        )
        pygame.draw.lines(
            self.surf,
            (255, 0, 0),
            False,
            [agent_pos.astype(int), pointr.astype(int)],
            2,
        )

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Clean up the environment
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
