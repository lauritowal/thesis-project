import gym
import gym_jsbsim

env = gym.make(ENV_ID)
env.reset()
state, reward, done, info = env.step(action)