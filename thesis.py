import gym
import gym_jsbsim
from gym_jsbsim.plotter_walt import PlotterWalt
import numpy as np

env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-NoFG-v0')
# env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-FG-v0')
env.reset()

pitch_data = []
roll_data = []
time_steps = []
time_step = 0
minutes = 3

def in_seconds(minutes):
    return minutes * 60

while time_step <= in_seconds(minutes=2):
    # env.render() # comment all render() for faster training
    # env.render("flightgear")
    # print("env.action_space.sample()", env.action_space.sample())
    (pitch_rad, roll_rad, time_step), _, done, _ = env.step(env.action_space.sample()) # take a random action

    roll_data.append(np.rad2deg(roll_rad))
    pitch_data.append(np.rad2deg(pitch_rad))
    time_steps.append(time_step)

plotter = PlotterWalt()
print(time_step)
plotter.plot(data=[roll_data, pitch_data], time=time_steps, titles=["roll", "pitch"])
print("done")