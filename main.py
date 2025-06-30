# FILE: main.py (or Jupyter Notebook)

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import custom classes and functions
from embedded_system_env import EmbeddedSystemEnv
from evaluation import evaluate_policy, performance_policy, powersave_policy, reactive_policy


# --- 1. Environment and Model Loading ---
# Create an environment instance for evaluation
env = EmbeddedSystemEnv()
# Optional: Check if the environment is valid
# check_env(env) 

# Load the pre-trained model
try:
    trained_model = PPO.load("ppo_power_manager")
except FileNotFoundError:
    print("Model file 'ppo_power_manager.zip' not found.")
    print("Please run train_agent.py first.")
    exit()

# --- 2. Policy Evaluation ---
print("Evaluating 'Performance' Policy...")
results_perf, _ = evaluate_policy(performance_policy, env)

print("\nEvaluating 'Power Save' Policy...")
results_save, _ = evaluate_policy(powersave_policy, env)

print("\nEvaluating 'Reactive' Policy...")
results_react, _ = evaluate_policy(reactive_policy, env)

# Define a policy function for the trained RL model
def rl_policy(observation):
    """Uses the trained model to predict an action."""
    action, _ = trained_model.predict(observation, deterministic=True)
    # Must convert the numpy array action to a standard int
    return int(action)

print("\nEvaluating 'RL Agent' Policy...")
results_rl, history_rl = evaluate_policy(rl_policy, env)


# --- 3. Visualization ---

# Plot 1: Comparative Bar Chart
all_results = {
    "Performance": results_perf,
    "Power Save": results_save,
    "Reactive": results_react,
    "RL Agent": results_rl,
}

policy_names = list(all_results.keys())
avg_power = [res["Avg Power"] for res in all_results.values()]
tasks_completed = [res["Total Tasks Completed"] for res in all_results.values()]
deadlines_missed = [res["Total Deadlines Missed"] for res in all_results.values()]

x = np.arange(len(policy_names))
width = 0.25

fig, ax1 = plt.subplots(figsize=(14, 8))

# Primary Y-Axis: Power Consumption
ax1.set_xlabel('Management Policy')
ax1.set_ylabel('Average Power Consumption', color='tab:blue', fontsize=12)
bars_power = ax1.bar(x - width, avg_power, width, label='Average Power', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.bar_label(bars_power, padding=3, fmt='%.1f', fontsize=9)

# Secondary Y-Axis: Tasks
ax2 = ax1.twinx()
ax2.set_ylabel('Number of Tasks', fontsize=12)
bars_completed = ax2.bar(x, tasks_completed, width, label='Tasks Completed', color='tab:green')
bars_missed = ax2.bar(x + width, deadlines_missed, width, label='Deadlines Missed', color='tab:red')
ax2.tick_params(axis='y')
ax2.bar_label(bars_completed, padding=3, fmt='%d', fontsize=9)
ax2.bar_label(bars_missed, padding=3, fmt='%d', fontsize=9)

# Dynamically set Y-axis limits
max_task_value = max(max(tasks_completed), max(deadlines_missed))
ax1.set_ylim(0, max(avg_power) * 1.15)
ax2.set_ylim(0, max_task_value * 1.15)

fig.suptitle('Power Management Policies Comparison', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(policy_names, rotation=10)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

fig.tight_layout()
plt.show()


# Plot 2: Dynamic Behavior of the RL Agent
history = history_rl
steps = range(len(history['power']))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
fig.suptitle('Dynamic Behavior of the RL Agent over one Episode', fontsize=16)

ax1.plot(steps, history['power'], color='blue', label='Power Consumption')
ax1.set_ylabel('Power')
ax1.legend()
ax1.grid(True)

ax2.plot(steps, history['temp'], color='red', label='Temperature (°C)')
ax2.axhline(y=95.0, color='darkred', linestyle='--', label='Critical Temp. Threshold')
ax2.set_ylabel('Temperature')
ax2.legend()
ax2.grid(True)

ax3.plot(steps, history['queue'], color='green', label='Task Queue Size')
ax3.set_xlabel('Simulation Step')
ax3.set_ylabel('Number of Tasks')
ax3.legend()
ax3.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()