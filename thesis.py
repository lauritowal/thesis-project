import gym
import gym_jsbsim
from gym_jsbsim.plotter_walt import PlotterWalt
import numpy as np


# env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-NoFG-v0')
env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-FG-v0')
env.reset()

pitch_data = []
roll_data = []
heading_data = []
time_steps = []
time_step = 0
minutes = 3

def in_seconds(minutes):
    return minutes * 60

action = np.array([0])
while time_step <= in_seconds(minutes=2):
    env.render() # comment render() for faster training
    # print("env.action_space.sample()", env.action_space.sample())

    # sample = env.action_space.sample()
    state, reward, done, _ = env.step(action) # take a random action
    aircraft_state, target_information, time_step = state

    action = np.array([target_information["relative_bearing_deg"]])

    print("aircraft_state", aircraft_state)
    print("target_information", target_information)
    print("time_step", time_step)

    '''roll_data.append(np.rad2deg(roll_rad))
    pitch_data.append(np.rad2deg(pitch_rad))
    time_steps.append(time_step)
    heading_data.append(heading_deg)'''

#plotter = PlotterWalt()
#print(time_step)
#plotter.plot(data=[roll_data, pitch_data, heading_data], time=time_steps, titles=["roll", "pitch", "heading"])
#print("done")