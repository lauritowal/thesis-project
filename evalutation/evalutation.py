import gym
from gym_jsbsim.agents import RandomAgent
from gym_jsbsim.normalise_env import NormalizeStateEnv
import numpy as np

def calculate_baseline_cumulative_reward():
    env = gym.make(id='guidance-v0',
                   jsbsim_path="/Users/walter/thesis_project/jsbsim",
                   max_episode_time_s=60,
                   flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")

    env = NormalizeStateEnv(env=env)
    agent = RandomAgent(action_space=env.action_space)

    rewards_history = []
    episode_length_history = []
    for _ in range(1000):
        env.reset()
        sum_reward = 0
        t = 0
        while True:
            action = agent.act()
            state, reward, done, info = env.step(action)
            sum_reward += reward
            t += 1
            if done:
                rewards_history.append(sum_reward)
                episode_length_history.append(t)
                break
    # print("rewards_history", rewards_history)
    # print("episode_length_history", episode_length_history)
    print(f"baseline mean cumulative reward: {np.mean(rewards_history)}")  # baseline cumulative reward: -1e+03
    print(f"baseline mean episode length: {np.mean(episode_length_history)}")

    return rewards_history, episode_length_history


calculate_baseline_cumulative_reward()

