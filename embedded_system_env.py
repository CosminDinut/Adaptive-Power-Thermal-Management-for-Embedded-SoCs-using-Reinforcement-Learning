# FILE: embedded_system_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class EmbeddedSystemEnv(gym.Env):
    """
    Custom Gymnasium Environment for simulating an embedded system
    that needs to manage power consumption vs. performance.
    """
    # Metadata for rendering
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(EmbeddedSystemEnv, self).__init__()

        # --- 1. Define the ACTION Space ---
        # The agent can choose one of 4 power states.
        # 0: Power Save (Minimum frequency, 1 core)
        # 1: Normal (Medium frequency, 2 cores)
        # 2: Performance (High frequency, 4 cores)
        # 3: Max Performance (Maximum frequency, 4 cores)
        self.action_space = spaces.Discrete(4)
        
        # Dictionary to map the action to physical parameters
        self.POWER_MODES = {
            0: {'performance': 10, 'power': 15, 'name': 'PowerSave'}, # Performance/Power per step
            1: {'performance': 30, 'power': 40, 'name': 'Normal'},
            2: {'performance': 60, 'power': 75, 'name': 'Performance'},
            3: {'performance': 90, 'power': 100, 'name': 'MaxPerf'}
        }

        # --- 2. Define the OBSERVATION (State) Space ---
        # The agent observes 3 parameters to make a decision:
        # - Task Queue Size (0 - 100)
        # - Current Temperature (20.0 - 100.0 degrees Celsius)
        # - Time since last task arrival (0 - 50 steps)
        self.observation_space = spaces.Box(
            low=np.array([0, 20.0, 0]), 
            high=np.array([100, 100.0, 50]), 
            dtype=np.float32
        )

        # --- 3. Simulation Parameters ---
        self.MAX_QUEUE_SIZE = 100
        self.MAX_TEMP = 95.0       # Critical temperature
        self.COOLING_FACTOR = 0.98 # How fast the system cools down
        self.HEATING_FACTOR = 0.03 # How fast it heats up per unit of power
        self.TASK_ARRIVAL_PROB = 0.3 # Probability of a new task arriving each step
        self._max_episode_steps = 500 # An episode lasts for 500 simulation steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the system's internal state at the beginning of each episode
        self.task_queue = [] # Task queue is empty
        self.current_temp = 30.0 # Initial temperature
        self.time_since_last_task = 0
        self.current_step = 0

        # Create the initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.current_step += 1
        
        # 1. Apply action and calculate performance/power for this step
        mode = self.POWER_MODES[action]
        performance_this_step = mode['performance']
        power_this_step = mode['power']
        
        tasks_completed_this_step = 0
        deadlines_missed_this_step = 0
        
        # 2. Process tasks in the queue (FIFO)
        if self.task_queue:
            self.task_queue[0]['work_left'] -= performance_this_step
            if self.task_queue[0]['work_left'] <= 0:
                tasks_completed_this_step += 1
                self.task_queue.pop(0) # Task finished, remove it

        # Decrement deadline counters for all pending tasks
        for task in self.task_queue:
            task['deadline'] -= 1
            if task['deadline'] < 0:
                deadlines_missed_this_step += 1
        # Remove tasks that have missed their deadline
        self.task_queue = [t for t in self.task_queue if t['deadline'] >= 0]

        # 3. Generate new tasks
        task_arrived = False
        if random.random() < self.TASK_ARRIVAL_PROB and len(self.task_queue) < self.MAX_QUEUE_SIZE:
            task_arrived = True
            new_task = {
                'work_left': random.randint(50, 200), # Task complexity
                'deadline': random.randint(5, 15)      # Steps until deadline
            }
            self.task_queue.append(new_task)
        
        # Update the time since the last task arrived
        if task_arrived:
            self.time_since_last_task = 0
        else:
            self.time_since_last_task += 1
            
        # 4. Update the temperature
        self.current_temp = (self.current_temp * self.COOLING_FACTOR) + (power_this_step * self.HEATING_FACTOR)
        self.current_temp = np.clip(self.current_temp, 20.0, 100.0)

        # 5. Calculate the REWARD (critical part!)
        reward = 0
        reward += tasks_completed_this_step * 50.0  # High reward for completing tasks
        reward -= deadlines_missed_this_step * 75.0 # Large penalty for missing deadlines
        reward -= power_this_step * 0.1             # Small, continuous penalty for power consumption
        overheated = self.current_temp > self.MAX_TEMP
        if overheated:
            reward -= 200.0 # Huge penalty for overheating

        # 6. Check termination conditions
        # Must explicitly cast to Python bool, as numpy.bool_ is not accepted by Gymnasium
        terminated = bool(overheated)
        truncated = self.current_step >= self._max_episode_steps

        # 7. Prepare data to be returned
        observation = self._get_obs()
        info = self._get_info()
        
        # Store metrics in 'info' for later evaluation
        info['power'] = power_this_step
        info['tasks_completed'] = tasks_completed_this_step
        info['deadlines_missed'] = deadlines_missed_this_step
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Assemble the observation vector
        return np.array([len(self.task_queue), self.current_temp, self.time_since_last_task], dtype=np.float32)

    def _get_info(self):
        # Assemble a dictionary with useful information
        return {
            "queue_size": len(self.task_queue),
            "temperature": self.current_temp
        }
    
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step: {self.current_step}, Queue: {len(self.task_queue)}, Temp: {self.current_temp:.2f}°C")