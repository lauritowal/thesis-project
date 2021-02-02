import gym
import gym_jsbsim
from gym_jsbsim.services.plotter import Map3DPlotter
import numpy as np
from gym.wrappers import  FlattenObservation

env = gym.make(id='Guidance-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")

state= env.reset()
print("state", state)


time_steps = []
aircraft_geo_lats = []
aircraft_geo_longs = []
aircraft_altitudes = []

aircraft_v_easts = []
aircraft_v_norths = []
aircraft_v_downs = []

track_angles = []
rewards = []
bounds: [gym_jsbsim.utils.GeoPosition] = []
time_step = 0
minutes = 3

def in_seconds(minutes: int) -> int:
    return minutes * 60

action = np.array([0])

print("start...")
while time_step <= in_seconds(minutes=5):
    # env.render("flightgear") # comment render for faster training

    observation = env.step(action)
    print("observation", observation)

    state, reward, done, info = observation

    target_heading_deg = state["target_heading_deg"]
    action = np.array([target_heading_deg])

    # diagram stuff
    aircraft_geo_longs.append(info["aircraft_long_deg"])
    aircraft_geo_lats.append(info["aircraft_lat_deg"])
    aircraft_altitudes.append(info["altitude_sl_ft"])

    aircraft_v_easts.append(info["aircraft_v_east_fps"])
    aircraft_v_norths.append(info["aircraft_v_north_fps"])
    aircraft_v_downs.append( - info["aircraft_v_down_fps"]) # TODO: Sure it is minus here?

    track_angles.append(state["aircraft_track_angle_deg"])
    rewards.append(reward)
    time_steps.append(time_step)

    time_step = info["time_step"] + 1

print("done")
print("plot...")

bound_points = []
for point in info["bounds"].values():
    lat, long = point.to_array()
    bound_points.append([long, lat])
bound_points.append(bound_points[0]) # append first point twice for plotting... find a more elegant solution later...

Map3DPlotter().plot(long=aircraft_geo_longs,
                    lat=aircraft_geo_lats,
                    altitude=aircraft_altitudes,
                    v_downs=aircraft_v_downs,
                    v_easts=aircraft_v_easts,
                    v_norths=aircraft_v_norths,
                    time=time_steps,
                    rewards=rewards,
                    track_angles=track_angles,
                    target_lat_deg=info["target_lat_deg"],
                    target_long_deg=info["target_long_deg"],
                    target_altitude_ft=info["target_altitude_ft"],
                    bounds=zip(*bound_points))
# '''
