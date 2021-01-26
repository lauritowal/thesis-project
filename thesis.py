import gym
import gym_jsbsim
from gym_jsbsim.plotter_walt import PlotterWalt
import numpy as np


# env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-NoFG-v0')
env = gym.make(id='JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-FG-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               flightgear_path="/Users/walter/thesis_project/FlightGear.app/Contents/MacOS")
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
    # env.render() # comment render() for faster training
    # print("env.action_space.sample()", env.action_space.sample())

    # sample = env.action_space.sample()
    state, reward, done, _ = env.step(action) # take a random action
    aircraft_state, target_information, wind_information, time_step = state

    # action = np.array([target_information["target_heading_deg"]])
    action = np.array([40])

#plotter = PlotterWalt()
#print(time_step)
#plotter.plot(data=[roll_data, pitch_data, heading_data], time=time_steps, titles=["roll", "pitch", "heading"])
#print("done")