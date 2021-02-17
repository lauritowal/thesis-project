import gym
from gym_jsbsim.agents import RandomAgent
from gym_jsbsim.normalise_env import NormalizeStateEnv


class Evalutation():
    def calculate_baseline_cumulative_reward(self):
        env = gym.make(id='guidance-v0',
                       jsbsim_path="/Users/walter/thesis_project/jsbsim",
                       max_episode_time_s=60,
                       flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")

        env = NormalizeStateEnv(env=env)
        agent = RandomAgent(action_space=env.action_space)

        history = []
        for _ in range(1000):
            sum_reward = self._run_one_episode(env, agent)
            history.append(sum_reward)
        avg_sum_reward = sum(history) / len(history)
        print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))  # baseline cumulative reward: -1e+03

    def _run_one_episode(self, env, agent):
        env.reset()
        sum_reward = 0
        while True:
            action = agent.act()
            state, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                break
        return sum_reward