import gym
from gym_jsbsim.wrappers.normalise_observation import NormalizeObservation
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3 import TD3
import numpy as np

# Parallel environments
env_kwargs = {
    "jsbsim_path": "/Users/walter/thesis_project/jsbsim",
    "max_episode_time_s":60
}

# env = make_vec_env('guidance-v0',
#                    n_envs=4,
#                    env_kwargs=env_kwargs)

env = gym.make('guidance-v0', **env_kwargs)
env = NormalizeObservation(env=env)

checkpoint_callback = CheckpointCallback(save_freq=1000,
                                         save_path='/Users/walter/thesis_project/data/checkpoints',
                                         name_prefix='rl_model')


eval_callback = EvalCallback(env,
                             best_model_save_path='./data/best_model',
                             log_path='./data/results',
                             eval_freq=500,
                             deterministic=True,
                             render=True)

# Create custom callback for showing videos similar to the one for rllib
max_episodes_callback = StopTrainingOnMaxEpisodes(max_episodes=30, verbose=1)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = TD3.load(path="./data/checkpoints/rl_model_500000_steps.zip", env=env, verbose=1)
model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)

# model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
# model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(1e10), callback=[checkpoint_callback, eval_callback, max_episodes_callback])

# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading