import gym
from gym_jsbsim.agents.agents import PerfectAgent
from gym_jsbsim.normalise_env import NormalizeStateEnv
import numpy as np
import os
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
    # agent = RandomAgent(action_space=env.action_space)
    agent = PerfectAgent(env=env)
    rewards = 0
    t = 0
    print("start...")

    while True:
        state, reward, done, info = env.step(action)
        aircraft_track_angle_rad, heading_to_target_rad, current_distance_to_target_km, runway_angle_rad = state

        print("heading_to_target_rad", heading_to_target_rad)

        # action = np.array([heading_to_target_rad])
        action = agent.act()

        print("action", action)

        images.append(env.render("rgb_array"))
        rewards += reward

        print("reward", reward)

        t += 1

        if done:
            print("###########################")
            print(f"done episode: {episode_counter}")
            print(f"episode time steps: {t}")
            print(f"total reward: {rewards}")
            print("###########################")

            if len(images) > 0:
                im = Image.fromarray(images[-1])
                im.save(f'{ROOT_DIR}/data/images/episode_{episode_counter}.png')

            break

print("done all episodes")
# """
