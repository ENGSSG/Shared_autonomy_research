import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as pyplot
from .bycycle_model import BicycleModel

class SharedAutonomyEnc(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 100}

    def __init__(self):
        super(SharedAutonomyEnv, self).__init()

        self.physics = BicycleModel(dt=0.01) # 100Hz physics update
        # Action Space: [steering (-1 to 1), throttle (-1 to 1)]
        # map these to physical limits in the step function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        #Observation Space : [x, y, heading, velocity, lane_offset, target_dist]
        # Using a large box for coordinates, limited for others
        low = np.array([-np.inf, -np.inf, -np.pi, -5.0, -10.0, -np.inf])
        high = np.array([np.inf, np.inf, np.pi, 20, 10.0, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # State initialization
        self.state = None
        self.target_lane_y = 0.0 # Standard lane center at y=0

        # Rendering 
        self.fig, self.ax = None, None
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        #Initialize state" [x=0, y=random_offset, psi=0, v=0]
        initial_y= self.np_random.uniform(-0.5, 0.5)
        self.state = np.array([0.0, initial_y, 0.0, 0.0])

        #Reset goal (just driving forward for now)
        self.target_lane_y = 0.0

        return self._get_obs(), {}

    def step(self, action):
        # 1. Map normalized action to physical constraints
        steer_cmd = action[0] * self.physics.max_steer
        accel_cmd = action[1] * self.physics.max_accel

        # 2. Physics Step
        self.state = self.physics.kinematics(self.state, [steer_cmd, accel_cmd])

        # 3. Calculate Observation features
        obs = self._get_obs()
        lane_offset = obs[4]

        # 4. Calculate Reward (Simple Lane Keeping for Phase 1 start)
        # Penalize lane deviation and control effor, reward speed
        reward = 1.0 * self.state[3] - 10.0 * (lane_offset**2) - 0.1 * np.sum(np.square(action))

        # 5. Check Done
        terminated = False
        truncated = False

        # Collision/Out of bounds check (Lane width approx 4, -> +/- 2m)
        if abs(self.state[1]) > 3.0:
            terminated = True
            reward -= 100.0 # Crash Penalty

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        x, y, psi, v = self.state
        lane_offset = y - self.target_lane_y
        #Simple target distance (arbitrary goal 100m ahead)
        target_dist = 100.0 - x

        return np.array([x, y, psi, v, lane_offset, target_dist], dtype=np.float32)

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plot.subplots()
            self.ax.set_xlim(-10,100)
            self.ax.set_ylim(-10,10)
            self.ax.set_aspect('equal')
            self.vehicle_plot, = self.ax.plot([], [], 'bo', markersize=10)
            self.lane_lines = [
                self.ax.axhline(y=2, color='k', linestyle='--'),
                self.ax.axhline(y=-2, color='k', linestyle='--')
            ]
            self.vehicle_plot.set_data([self.state[0]], [self.state[1]])
            self.ax.set_xlim(self.state[0] - 10, self.state[0] + 20)

            plt.draw()
            plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close()