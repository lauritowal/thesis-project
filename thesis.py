import gym
import gym_jsbsim
from gym_jsbsim.agents import RandomAgent
from gym_jsbsim.environment import GuidanceEnv
from gym_jsbsim.normalise_env import NormalizeStateEnv
from gym_jsbsim.services.plotter import MapPlotter
import numpy as np
import os
from gym_jsbsim.utils import in_seconds
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
NUM_EPISODES = 1

env = gym.make(id='guidance-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               max_episode_time_s=1,
               flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")

env = NormalizeStateEnv(env=env)
env.reset()
image = env.render("rgb_array")
# plt.imshow(image)
# plt.show()

for episode_counter in range(NUM_EPISODES):
    state = env.reset() # TODO: Common practice to get also info in reset?
    # print("state at start", state)
    sim_time_steps = []
    aircraft_geo_lats = []
    aircraft_geo_longs = []
    aircraft_altitudes = []

    aircraft_v_easts = []
    aircraft_v_norths = []
    aircraft_v_downs = []
    images = []
    track_angles = []
    rewards = []
    # bounds: [gym_jsbsim.utils.GeoPosition] = []
    sim_time_step = 0
    t = 0

    action = np.array([0])

    print("start...")

    done_counter = 0
    images.append(env.render("rgb_array"))

    agent = RandomAgent(action_space=env.action_space)

    while True:
        state, reward, done, info = env.step(action)

        # print("state", state)

        ground_speed, aircraft_track_angle_deg, heading_to_target_deg, current_distance_to_target_m = state

        # env.render('flightgear')

        images.append(env.render("rgb_array"))

        # action = np.array([heading_to_target_deg])

        action = agent.act()
        print(action)
        # diagram stuff
        aircraft_geo_longs.append(info["aircraft_long_deg"])
        aircraft_geo_lats.append(info["aircraft_lat_deg"])
        aircraft_altitudes.append(info["altitude_sl_ft"])

        aircraft_v_easts.append(info["aircraft_v_east_fps"])
        aircraft_v_norths.append(info["aircraft_v_north_fps"])
        aircraft_v_downs.append(- info["aircraft_v_down_fps"])  # TODO: Sure it is minus here?

        track_angles.append(aircraft_track_angle_deg)
        rewards.append(reward)
        sim_time_steps.append(sim_time_step)

        print("reward", reward)

        # sim_time_step = info["time_step"] + 1
        t += 1
        if done:
            print("###########################")
            print(f"done episode: {episode_counter}")
            print(f"episode time steps {t}")
            print("simulation time", sim_time_step)
            print("rewards", np.sum(rewards))
            # print("state", state)
            print("###########################")

            # video_file_name = f'{ROOT_DIR}/data/videos/episode_{episode_counter}'
            # MapPlotter.convert2video(images=images, file_name=video_file_name)
            if len(images) > 0:
                im = Image.fromarray(images[-1])
                im.save(f'{ROOT_DIR}/data/images/episode_{episode_counter}.png')

            # gifs_file_name = f'{ROOT_DIR}/data/gifs/episode_{episode_counter}'
            # MapPlotter().convert2gif(images=images, file_name=gifs_file_name) # Cleanup...

            # htmls_file_name = f'{ROOT_DIR}/data/htmls/episode_{episode_counter}'
            # MapPlotter().plot(long=aircraft_geo_longs,
            #                   lat=aircraft_geo_lats,
            #                   altitude=aircraft_altitudes,
            #                   v_downs=aircraft_v_downs,
            #                   v_easts=aircraft_v_easts,
            #                   v_norths=aircraft_v_norths,
            #                   time=sim_time_steps,
            #                   rewards=rewards,
            #                   track_angles=track_angles,
            #                   target_lat_deg=info["target_lat_deg"],
            #                   target_long_deg=info["target_long_deg"],
            #                   target_altitude_ft=info["target_altitude_ft"],
            #                   bounds =zip(*bound_points),
            #                   file_name=htmls_file_name,
            #                   show=True)
            break

print("done")
print("plot...") # for testing
# """
