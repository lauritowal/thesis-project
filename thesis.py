import gym
import gym_jsbsim
from gym_jsbsim.plotter_walt import PlotterWalt
import numpy as np

# env = gym.make('JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-NoFG-v0')
env = gym.make(id='JSBSim-GuidanceTask-Cessna172P-Shaping.STANDARD-FG-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")
state = env.reset()

print("env.action_space.sample()", env.action_space.sample())
print("env.observation_space.sample()", env.observation_space.sample())

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
    env.render()
    # sample = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    target_heading_deg = state["target_heading_deg"]
    action = np.array([target_heading_deg])

