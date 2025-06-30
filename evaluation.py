# FILE: evaluation.py

def evaluate_policy(policy, env, episodes=5):
    """
    Runs multiple episodes using a given policy and returns aggregated metrics.
    
    :param policy: A function that takes an observation and returns an action.
    :param env: The Gymnasium environment to run the simulation in.
    :param episodes: The number of episodes to average the results over.
    :return: A dictionary with aggregated results and a history dict for one episode.
    """
    total_power = 0
    total_tasks_completed = 0
    total_deadlines_missed = 0
    
    # For plotting, we store the data from a single episode
    history = {'power': [], 'temp': [], 'queue': []}

    for i in range(episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Aggregate metrics
            total_power += info['power']
            total_tasks_completed += info['tasks_completed']
            total_deadlines_missed += info['deadlines_missed']
            
            # Store data for the history plot (only for the first episode)
            if i == 0:
                history['power'].append(info['power'])
                history['temp'].append(info['temperature'])
                history['queue'].append(info['queue_size'])

    num_steps = env._max_episode_steps * episodes
    results = {
        "Avg Power": total_power / num_steps,
        "Total Tasks Completed": total_tasks_completed,
        "Total Deadlines Missed": total_deadlines_missed
    }
    return results, history

# --- Define Baseline Policies ---

def performance_policy(observation):
    """Always chooses the 'Max Performance' action."""
    return 3

def powersave_policy(observation):
    """Always chooses the 'Power Save' action."""
    return 0

def reactive_policy(observation):
    """A simple, rule-based policy that reacts to the queue size."""
    queue_size = observation[0]
    if queue_size > 10:
        return 3 # Max Performance
    elif queue_size > 3:
        return 1 # Normal
    else:
        return 0 # Power Save