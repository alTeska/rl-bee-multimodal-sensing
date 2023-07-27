import numpy as np
import pygame
import gymnasium as gym
from pygame.locals import *
from gymnasium import spaces


def cone_locations(
    agent_location, agent_theta, cone_phi, scale_factor
) -> tuple[np.ndarray, np.ndarray]:
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


def intersect_segments(sA, sB) -> tuple[float, float]:
    """Return an intersection point for two line segments
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection?useskin=vector#Given_two_points_on_each_line_segment

    Args:
        sA: segment A
        sB: segment B
    """
    pA1, pA2 = sA
    pB1, pB2 = sB

    x1, y1 = pA1
    x2, y2 = pA2
    x3, y3 = pB1
    x4, y4 = pB2

    div = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if div == 0:
        return (None, None)

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (div)
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (div)

    if (t < 0.0) or (t > 1.0):
        return (None, None)

    if (u < 0.0) or (u > 1.0):
        return (None, None)

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


def intersect_segments(segment1, segment2) -> tuple[float, float]:
    """
    Find the intersection point for two line segments.

    The function implements the formula for the intersection point of two line segments.
    For more details, refer to: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection?useskin=vector#Given_two_points_on_each_line_segment

    Args:
        segment1 (tuple): A tuple containing two points (x, y) representing the first line segment.
        segment2 (tuple): A tuple containing two points (x, y) representing the second line segment.

    Returns:
        tuple: A tuple (px, py) representing the intersection point (x, y).
              If the two line segments do not intersect, returns (None, None).
    """
    pA1, pA2 = segment1
    pB1, pB2 = segment2

    x1, y1 = pA1
    x2, y2 = pA2
    x3, y3 = pB1
    x4, y4 = pB2

    div = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if div == 0:
        return (None, None)

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / div
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / div

    if (t < 0.0) or (t > 1.0):
        return (None, None)

    if (u < 0.0) or (u > 1.0):
        return (None, None)

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


class BeeWorld(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        size=10,
        dt=0.1,
        render_mode="human",
        max_episode_steps=1000,
        goal_size=2.0,
        walls=[
            [(5.0, 0.0), (5.0, 5.0)],
        ],
        agent_location_range=[(0.0, 2.0), (0.0, 10.0)],
        goal_location_range=[(5.0, 10.0), (0.0, 10.0)],
        noise_smell=False,
        noise_vision=False,
    ):
        self.dtype = "float32"
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.agent_location_range = agent_location_range
        self.goal_location_range = goal_location_range
        self.noise_vision = noise_vision
        self.noise_smell = noise_smell
        self.steps = 0

        self.size = size  # Room size
        self.dt = dt  # Integration timestep
        self._agent_vel = 0.0  # Translational velocity
        self._agent_theta = 0.0  # Agent's direction as angle from x-axis
        self._agent_ang_vel = 0.0  # Angular velocity

        self.linear_acceleration_range = 0.5
        self.angular_acceleration_range = 0.1

        self.cone_phi = np.pi / 8  # Vision cone angle

        self.goal_size = goal_size

        self.walls = [
            [(0.0, 0.0), (0.0, self.size)],
            [(0.0, 0.0), (self.size, 0.0)],
            [(0.0, self.size), (self.size, self.size)],
            [(self.size, 0.0), (self.size, self.size)],
        ] + walls

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Discrete(2),
                "smell": spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                "velocity": spaces.Box(-1, 1, shape=(2,), dtype=self.dtype),
                "time": spaces.Box(
                    low=0,
                    high=1,
                    dtype=self.dtype,
                ),
                "wall": spaces.Box(0, 1, dtype=self.dtype),
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
        self.font = None

        self.trajectory = []

        self.obs = None

    def _check_vision(self) -> int:
        """
        Check if the agent can see the goal and return the result.

        The function determines if the agent has a direct line of sight to the goal location.
        It computes the angle between the line connecting the agent's location and the goal location,
        and the agent's orientation angle. If the absolute difference between these angles is greater
        than the field of vision (cone_phi), the goal is not visible.

        If the noise_vision flag is True, there is a chance of random noise being added to the result.

        Returns:
            int: Returns 1 if the agent can see the goal, and 0 otherwise. The return value is influenced
                by the noise_vision flag (if set, there is a chance of adding random noise to the result).
        """
        noise = 0
        if self.noise_vision:
            noise = np.random.binomial(1, 0.5)

        ray = self._target_location - self._agent_location  # raycast from agent to goal
        ang = (np.arctan2(ray[1], ray[0])) % (2 * np.pi)  # angle of raycast

        diff = np.abs(ang - self._agent_theta)

        if (diff > self.cone_phi) and ((2 * np.pi - diff) > self.cone_phi):
            return 0 + noise

        if self.segment_wall_intersections(
            [self._agent_location, self._target_location]
        ):
            return 0 + noise

        return 1 - noise

    def _get_smell(self) -> np.ndarray:
        """
        Calculate the strength of the smell at the agent's current location.

        The function calculates the smell strength at the agent's current location as the exponential decay
        of the distance between the agent's location and the goal location. The distance is computed using
        the L2 norm (Euclidean distance) between the two points.

        If the noise_smell flag is True, there is a chance of adding random noise to the smell strength.

        Returns:
            np.ndarray: An array containing the strength of the smell at the agent's current location.
                        The value is calculated as the exponential decay of the distance between the agent
                        and the goal, and it may be influenced by random noise (if noise_smell is True).
        """
        noise = 0
        if self.noise_smell:
            noise = np.random.normal(0.5, 0.2)

        # Calculate the exponential decay of the distance between agent and goal as smell strength
        smell_strength = np.array(
            [
                np.exp(
                    -np.linalg.norm(self._agent_location - self._target_location, ord=2)
                )
                + noise
            ],
            dtype=self.dtype,
        )

        return smell_strength

    def _get_time(self) -> np.ndarray:
        """Get the current timestep scaled between 0 and 1."""
        self.steps += 1
        return np.array([self.steps / self.max_episode_steps], dtype=self.dtype)

    def _get_visible_wall(self, n_casts=7) -> np.ndarray:
        """
        Get the distance to the closest wall within the agent's cone of vision using raycasts. The function performs raycasts equally spaced inside the agent's field of vision (cone_phi) to detect
        the distance to the closest wall in the environment.

        Args:
            n_casts (int, optional): Number of raycasts to perform. Defaults to 7.

        Returns:
            np.ndarray: An array containing the distance to the closest wall in the agent's cone of vision.
                        The distance is normalized by the agent's size (self.size) to provide a relative measure.
        """
        mins = []

        angles = np.linspace(-self.cone_phi, self.cone_phi, n_casts)

        for angle in angles:
            ray_point = self._agent_location + [
                2 * self.size * np.cos(self._agent_theta + angle),
                2 * self.size * np.sin(self._agent_theta + angle),
            ]

            ints = self.segment_wall_intersections([self._agent_location, ray_point])
            assert ints

            ds = [
                np.linalg.norm(np.array(i) - self._agent_location, ord=2) for i in ints
            ]

            mins.append(min(ds))
        return np.array([min(mins) / self.size], dtype=self.dtype)

    def _get_obs(self) -> dict:
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
            "wall": self._get_visible_wall(),
        }

    def _get_info(self) -> dict:
        """
        Provides auxiliary information
        """
        return {"is_success": False}

    def _check_goal_intersections(self) -> bool:
        """
        Check for intersections between the goal and walls in the environment.

        The function iterates through all the walls in the environment and checks if there are any intersections
        between the goal and the walls. An intersection can occur in the following cases:
        1. If one of the endpoints of a wall is within the goal region, an intersection exists.
        2. If the distance between the goal center and the projection of the goal onto the wall is less than the goal size,
        an intersection exists.

        Returns:
            bool: True if there is an intersection between the goal and walls, False otherwise.
        """
        for wall in self.walls:
            # if one of the endpoints of a wall is in the goal they intersect
            if np.linalg.norm(self._target_location - wall[0], ord=2) < self.goal_size:
                return True
            if np.linalg.norm(self._target_location - wall[1], ord=2) < self.goal_size:
                return True

            wall_vector = np.array(wall[1]) - wall[0]
            target_vector = self._target_location - wall[0]
            a = np.linalg.norm(target_vector, ord=2)
            c = np.linalg.norm(wall_vector, ord=2)

            # project a vector between one endpoint and the goal center on the wall
            projection = np.dot(target_vector, wall_vector) / c

            # if the projection is negative or larger than the wall there are no intersections
            if (projection > c) or (projection < 0):
                continue

            dist = np.sqrt(a**2 - projection**2)

            # check the distance beween the goal and the wall segment
            if dist < self.goal_size:
                return True

        return False

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment to start a new episode.

        This function resets the environment to start a new episode. It sets the initial state of the agent,
        including its location, velocity, direction, and angular velocity. It also initializes the goal location.

        Args:
            seed (int or None): Seed to initialize the random number generator for reproducibility. Defaults to None.
            options (dict or None): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple[dict, dict]: A tuple containing the initial observation and info dictionaries after the reset.
        """
        super().reset(seed=seed)

        self.steps = 0
        self._agent_vel = 0.0  # Translational velocity
        self._agent_theta = 0.0  # Agent's direction as angle from x-axis
        self._agent_ang_vel = 0.0  # Angular velocity

        #  Agent location limited on x-axis
        self._agent_location = np.array(
            [
                self.np_random.uniform(
                    low=self.agent_location_range[0][0],
                    high=self.agent_location_range[0][1],
                    size=1,
                )[0],
                self.np_random.uniform(
                    low=self.agent_location_range[1][0],
                    high=self.agent_location_range[1][1],
                    size=1,
                )[0],
            ],
            dtype=self.dtype,
        )

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array(
                [
                    self.np_random.uniform(
                        low=self.goal_location_range[0][0],
                        high=self.goal_location_range[0][1],
                        size=1,
                    )[0],
                    self.np_random.uniform(
                        low=self.goal_location_range[1][0],
                        high=self.goal_location_range[1][1],
                        size=1,
                    )[0],
                ],
                dtype=self.dtype,
            )

            if self._check_goal_intersections():
                self._target_location = self._agent_location

        observation = self._get_obs()
        info = self._get_info()

        self.obs = observation
        self.trajectory = []  # Reset trajectory

        if self.render_mode == "human":
            self.render()

        return observation, info

    def segment_wall_intersections(self, segment) -> list:
        """
        Check whether a line segment intersects with any walls in the environment.

        The function iterates through all the walls in the environment and checks for intersections
        with the given line segment. It uses the 'intersect_segments' function to find intersection points.

        Args:
            segment (tuple): The line segment represented as a tuple of two points (start_point, end_point).

        Returns:
            list: A list of intersection points between the line segment and walls. If no intersections are found, an empty list is returned.
        """
        intersections = []

        for wall in self.walls:
            intersection_point = intersect_segments(segment, wall)
            if intersection_point[0] is not None:
                intersections.append(intersection_point)

        return intersections

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Perform a single time step in the environment based on the given action.

        The function updates the agent's state (position and velocity) based on the action,
        calculates the reward for the current step, and checks if the episode is terminated.

        Args:
            action (np.ndarray): The action to be taken by the agent.

        Returns:
            tuple[dict, float, bool, bool, dict]: A tuple containing the updated observation, the reward obtained from the action,
            a boolean indicating if the episode is terminated due to success, a boolean indicating if the episode is done, and an info dictionary.
        """
        reward = 0

        old_agent_location = self._agent_location.copy()

        # Update agent's velocity and angular velocity based on the action
        self._agent_vel += self.dt * action[0] * self.linear_acceleration_range
        self._agent_vel = np.clip(self._agent_vel, 0, 1)

        self._agent_ang_vel += self.dt * action[1] * self.angular_acceleration_range
        self._agent_ang_vel = np.clip(self._agent_ang_vel, -0.3, 0.3)

        # Update agent's direction (theta) based on the angular velocity
        self._agent_theta += self.dt * self._agent_ang_vel
        self._agent_theta = self._agent_theta % (2 * np.pi)

        # Update agent's location based on its velocity and direction
        self._agent_location += [
            self.dt * self._agent_vel * np.cos(self._agent_theta),
            self.dt * self._agent_vel * np.sin(self._agent_theta),
        ]

        # Check for collisions with walls
        if wall_intersections := self.segment_wall_intersections(
            [old_agent_location, self._agent_location]
        ):
            self._agent_location = old_agent_location
            self._agent_vel = 0
            reward -= 100

        # Calculate the distance between the agent and the goal
        goal_distance = np.linalg.norm(
            self._target_location - self._agent_location, ord=2
        )

        # Check if the episode is terminated (agent reaches the goal)
        terminated = goal_distance < self.goal_size

        # Get the updated observation after the step
        observation = self._get_obs()
        self.obs = observation

        # Calculate the reward for the current step
        factor = 0.01
        reward += 1000 if terminated else 0  # Binary sparse rewards
        reward -= 0.3 * np.sum(np.abs(action) ** 2)  # Energy expenditure
        reward -= goal_distance * factor

        # Create the info dictionary for the current step
        info = self._get_info()
        info["is_success"] = terminated

        # Append the agent's current location to the trajectory
        self.trajectory.append(self._agent_location.copy())

        if self.render_mode == "human":
            # Check for events in the Pygame window (if the rendering mode is "human")
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    quit()

            self.render()

        return observation, reward, terminated, terminated, info

    def render(self, scale=0.9):
        """
        Renders the current state of the environment using Pygame.

        The function creates a Pygame window and draws the agent, target, trajectory, walls, and cones of vision on the screen.

        Args:
            scale (float, optional): Scale factor for the rendering. Defaults to 0.9.

        Notes:
            - The screen is scaled to a 90% size to fit the objects within the window.
            - The agent's and target's positions are transformed based on the scale factor.

        Returns:
            If the rendering mode is 'human':
                None
            If the rendering mode is 'rgb_array':
                np.ndarray: An RGB array representing the rendered image.
        """
        if self.screen is None and self.render_mode == "human":
            # Initialize Pygame window if not already created
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("BeeWorld")
            self.font = pygame.font.SysFont("monospace", 10)

        self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill((255, 255, 255))

        # Calculate scaled screen properties
        screen_width = int(self.screen_size[0] * scale)
        screen_height = int(self.screen_size[1] * scale)
        screen_offset_x = int((self.screen_size[0] - screen_width) / 2)
        screen_offset_y = int((self.screen_size[1] - screen_height) / 2)
        scale_factor = screen_width / self.size

        # Calculate and draw agent's and target's positions
        agent_pos = self._agent_location * scale_factor
        agent_pos += np.array([screen_offset_x, screen_offset_y])
        target_pos = self._target_location * scale_factor
        target_pos += np.array([screen_offset_x, screen_offset_y])

        pygame.draw.circle(self.surf, (255, 0, 0), agent_pos.astype(int), 5)
        pygame.draw.circle(
            self.surf,
            (0, 255, 0),
            target_pos.astype(int),
            self.goal_size * scale_factor,
        )

        # Draw agent's trajectory
        if len(self.trajectory) > 1:
            trajectory_points = [
                pos * scale_factor + np.array([screen_offset_x, screen_offset_y])
                for pos in self.trajectory
            ]
            pygame.draw.lines(self.surf, (0, 0, 255), False, trajectory_points, 2)

        # Draw cones of vision
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

        # Draw walls
        for wall in self.walls:
            pygame.draw.lines(
                self.surf,
                (0, 0, 0),
                False,
                [
                    (
                        np.array(wall[0]) * scale_factor
                        + np.array([screen_offset_x, screen_offset_y])
                    ).astype(int),
                    (
                        np.array(wall[1]) * scale_factor
                        + np.array([screen_offset_x, screen_offset_y])
                    ).astype(int),
                ],
            )

        if self.render_mode == "human":
            assert self.screen is not None

            # Draw observation information on the Pygame window
            label = self.font.render(
                f"vision: {self.obs['vision']}; smell: {self.obs['smell'][0]:.4f}; wall: {self.obs['wall'][0]:.4f}",
                1,
                (0, 0, 0),
            )
            self.screen.blit(self.surf, (0, 0))
            self.screen.blit(label, (0, 5))

            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # Return the rendered image as an RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        """
        Clean up the environment
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
