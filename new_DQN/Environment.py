import math
import random

time_limit = 500          # seconds
penalty = -1000
border_width, border_height = 500, 500   # meters
waypoint_radius = 10      # meters
final_reward = 1000
max_velocity_mag = 30     # m/s

k = 0
b = 0
m = 0

ACTION_VALUES = list(range(-5, 6))  # [-5, -4, ..., 5]


class DiscreteActionSpace2D:
    """
    A discrete action space of 2D acceleration vectors.

    There are 11 values per axis, so total actions = 11 * 11 = 121.
    Each action is a tuple: (ax, ay).
    """
    def __init__(self, values):
        self.values = list(values)
        self.actions = [(ax, ay) for ax in self.values for ay in self.values]
        self.n = len(self.actions)
        self.rng = random.Random()

    def sample(self):
        """Return a random 2D action tuple."""
        return self.actions[self.rng.randrange(self.n)]

    def seed(self, seed=None):
        self.rng.seed(seed)


class ContinuousSpace:
    """A 6D state: [x, y, vx, vy, wx, wy]."""
    def __init__(self, n):
        self.n = n
        self.rng = random.Random()

    def sample(self):
        waypoint_x = self.rng.uniform(-0.9 * border_width / 2, 0.9 * border_width / 2)
        waypoint_y = self.rng.uniform(-0.9 * border_height / 2, 0.9 * border_height / 2)

        agent_x = self.rng.uniform(-0.5 * border_width / 2, 0.5 * border_width / 2)
        agent_y = self.rng.uniform(-0.5 * border_height / 2, 0.5 * border_height / 2)

        agent_v_mag = self.rng.uniform(0, 0.2 * max_velocity_mag)
        agent_v_angle = self.rng.uniform(0, 2 * math.pi)
        agent_vx = agent_v_mag * math.cos(agent_v_angle)
        agent_vy = agent_v_mag * math.sin(agent_v_angle)

        # Return the normalized index
        return [
            agent_x,
            agent_y,
            agent_vx,
            agent_vy,
            waypoint_x - agent_x,
            waypoint_y - agent_y,
        ]

    def seed(self, seed=None):
        self.rng.seed(seed)


class Environment:
    def __init__(self):
        self.observation_space = ContinuousSpace(6)
        self.action_space = DiscreteActionSpace2D(ACTION_VALUES)
        self.state = []
        self.time = 0

    def reset(self, seed=None):
        self.time = 0
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.state = self.observation_space.sample()
        return self.state.copy()

    def _parse_action(self, action):
        """
        Accept either:
        - an action index (int), or
        - a 2D action tuple/list [ax, ay]
        """
        if isinstance(action, int):
            return self.action_space.actions[action]

        if hasattr(action, "__len__") and len(action) == 2:
            return float(action[0]), float(action[1])

        raise TypeError("Action must be either an int index or a 2D (ax, ay) action.")

    def step(self, action, time):
        ax, ay = self._parse_action(action)
        self.time = time

        x, y, vx, vy, wx, wy = self.state
        dt = 0.1

        prev_distance = (wx * wx + wy * wy) ** 0.5

        # Kinematics update
        dx = vx * dt + 0.5 * ax * dt * dt
        dy = vy * dt + 0.5 * ay * dt * dt

        x += dx
        y += dy
        vx += ax * dt
        vy += ay * dt

        # Relative waypoint position updates because the waypoint is fixed
        wx -= dx
        wy -= dy

        self.state = [x, y, vx, vy, wx, wy]

        reward, terminated = self.acquire_reward(prev_distance, (ax, ay))
        return self.state.copy(), reward, terminated

    def acquire_reward(self, prev_distance, action):
        """Termination conditions: reach waypoint, hit wall, exceed time, exceed speed."""
        x, y, vx, vy, wx, wy = self.state

        # Current distance after the step
        curr_distance = (wx * wx + wy * wy) ** 0.5

        # Action penalty: keep control smooth
        ax, ay = action
        acc_penalty = 0.001 * (ax * ax + ay * ay)

        # Time penalty: encourage faster completion
        time_penalty = 0.01

        # Progress reward: positive if distance decreased
        progress_reward = 5.0 * (prev_distance - curr_distance)

        reward = progress_reward - time_penalty - acc_penalty

        # Terminal conditions
        if self.time >= time_limit:
            return reward - 50.0, True

        if (
                x >= border_width / 2 or x <= -border_width / 2 or
                y >= border_height / 2 or y <= -border_height / 2
        ):
            return reward - 50.0, True

        if vx * vx + vy * vy >= max_velocity_mag * max_velocity_mag:
            return reward - 50.0, True

        if curr_distance < waypoint_radius:
            return reward + 100.0, True

        return reward, False
