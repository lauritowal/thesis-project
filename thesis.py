import gym
import gym_jsbsim

env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-FG-v0')
env.reset()
for _ in range(1000):
    env.render('flightgear')
    env.step(env.action_space.sample()) # take a random action
env.close()