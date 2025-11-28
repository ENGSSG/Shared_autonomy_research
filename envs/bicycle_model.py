import numpy as np

class BicycleModel:
    def __init__(self, dt=0.01):
        # Vehicle Parameters (Traxxas Slash 1/10 scale approx)
        self.L = 0.33  # Wheelbase (m)
        self.lr = 0.165 # CG to rear axle
        self.lf = 0.165 # CG to front axle
        self.dt = dt   # Simulation step (0.01s = 100Hz)
        
        # Constraints
        self.max_steer = np.radians(30) # 30 degrees
        self.max_accel = 5.0 # m/s^2
        self.max_speed = 10.0 # m/s

    def kinematics(self, state, action):
        """
        Updates state based on kinematic bicycle model.
        State: [x, y, psi (heading), v (velocity)]
        Action: [steering_angle, acceleration]
        """
        x, y, psi, v = state
        steering, accel = action

        # Clip actions
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        accel = np.clip(accel, -self.max_accel, self.max_accel)

        # Calculate slip angle (beta)
        beta = np.arctan((self.lr / (self.lf + self.lr)) * np.tan(steering))

        # Update state (Euler integration)
        dx = v * np.cos(psi + beta)
        dy = v * np.sin(psi + beta)
        dpsi = (v / self.L) * np.sin(beta)
        dv = accel

        new_x = x + dx * self.dt
        new_y = y + dy * self.dt
        new_psi = psi + dpsi * self.dt
        new_v = v + dv * self.dt
        
        # Clip velocity (cannot go backwards faster than -2 m/s, or exceed max)
        new_v = np.clip(new_v, -2.0, self.max_speed)
        
        # Normalize heading to -pi to pi
        new_psi = np.arctan2(np.sin(new_psi), np.cos(new_psi))

        return np.array([new_x, new_y, new_psi, new_v])