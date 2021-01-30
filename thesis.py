import gym
from gym_jsbsim.services.plotter import Map3DPlotter
import numpy as np

env = gym.make(id='Guidance-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")
state = env.reset()

time_steps = []
aircraft_geo_lats = []
aircraft_geo_longs = []
aircraft_altitudes = []
track_angles = []
rewards = []
time_step = 0
minutes = 3

def in_seconds(minutes: int) -> int:
    return minutes * 60

action = np.array([0])

print("start...")
while time_step <= in_seconds(minutes=1):
    # env.render("flightgear") # comment render for faster training

    observation = env.step(action)
    print(observation)
    state, reward, done, _ = observation

    aircraft_geo_longs.append(state["aircraft_long_deg"])
    aircraft_geo_lats.append(state["aircraft_lat_deg"])
    aircraft_altitudes.append(state["altitude_sl_ft"])
    track_angles.append(state["track_angle_deg"])
    rewards.append(reward)

    target_heading_deg = state["target_heading_deg"]
    action = np.array([0])

    time_steps.append(time_step)
    time_step = state["time_step"] + 1

print("done")
print("plot...")

data = [
    aircraft_geo_longs,
    aircraft_geo_lats,
    aircraft_altitudes,
    time_steps,
    rewards,
    track_angles
]
Map3DPlotter().plot(data)
