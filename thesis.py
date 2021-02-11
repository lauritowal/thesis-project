import gym
import gym_jsbsim
from gym_jsbsim.agents import RandomAgent
from gym_jsbsim.environment import GuidanceEnv
from gym_jsbsim.normalise_env import NormalizeStateEnv
from gym_jsbsim.services.map_plotter import MapPlotter
import numpy as np
import os
from gym_jsbsim.utils import in_seconds
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
NUM_EPISODES = 1

env = gym.make(id='guidance-v0',
               jsbsim_path="/Users/walter/thesis_project/jsbsim",
               max_episode_time_s=60,
               flightgear_path="/Users/walter/FlightGear.app/Contents/MacOS/")

env = NormalizeStateEnv(env=env)
env.reset()
image = env.render("rgb_array")
# plt.imshow(image)
# plt.show()

for episode_counter in range(NUM_EPISODES):
    state = env.reset()

    images = []
    action = np.array([0])
    done_counter = 0
    images.append(env.render("rgb_array"))
    agent = RandomAgent(action_space=env.action_space)
    t = 0
    print("start...")

    while True:
        state, reward, done, info = env.step(action)

        aircraft_track_angle_deg, heading_to_target_deg, current_distance_to_target_km = state

        print("state", state)

        # env.render('flightgear')

        images.append(env.render("rgb_array"))

        action = agent.act()

        print("action", np.rad2deg(action))
        print("reward", reward)

        t += 1
        if done:
            print("###########################")
            print(f"done episode: {episode_counter}")
            print(f"episode time steps {t}")
            print("###########################")

            if len(images) > 0:
                im = Image.fromarray(images[-1])
                im.save(f'{ROOT_DIR}/data/images/episode_{episode_counter}.png')

            break

print("done all episodes")
# """
