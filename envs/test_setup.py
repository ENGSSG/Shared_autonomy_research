import time
import numpy as np
from vehicle_env import SharedAutonomyEnv

def test_environment():
    env = SharedAutonomyEnv()
    obs, _ = env.reset()

    print("Environment initialized")
    print(f"Initial State: {obs}")

    start_time = time.time()
    steps = 0

    # Simulate for 5 seconds (500 steps at 100Hz)

    try:
        for _ in range(6000):
            # Simple action: Accelerate and steer slightly left
            action = np.array([0.1, 0.5])

            obs, reward, done, _, _ = env.step(action)
            # Comment below code out for simulation without rendering
            env.render()
            

            steps += 1
            if done:
                print("Crashed!")
                env.reset()

    except KeyboardInterrupt:
        pass

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nSimulation Report:")
    print(f"Total Steps: {steps}")
    print(f"Real Time Elapsed: {duration:.2f}s")
    print(f"Simulated Frequency: {steps/duration:.2f}Hz (limited by render)")
    print(f"Final State: {obs}")

if __name__ == "__main__":
    test_environment()