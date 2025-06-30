# FILE: train_agent.py

from stable_baselines3 import PPO
from embedded_system_env import EmbeddedSystemEnv # Assuming the class is in this file

# Create an instance of the training environment
train_env = EmbeddedSystemEnv()

# We choose the PPO algorithm with a Multi-Layer Perceptron (Mlp) policy
# `verbose=1` will print training progress
model = PPO("MlpPolicy", train_env, verbose=1)

print("\nStarting RL Agent Training... (This may take a few minutes)")
# Train the model for a set number of timesteps
# 100,000 is a good starting point for this problem
model.learn(total_timesteps=100000)
print("Training finished!")

# Save the trained model to a file
model.save("ppo_power_manager")
print("Model saved to ppo_power_manager.zip")