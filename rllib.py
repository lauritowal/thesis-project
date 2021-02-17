from typing import Optional

import gym
import gym_jsbsim
from PIL import Image
from gym import Env
from gym_jsbsim.environment import GuidanceEnv
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ddpg import td3, TD3Trainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from gym_jsbsim.normalise_env import NormalizeStateEnv
from ray import tune
import numpy as np

def env_creator(env_config):
    return GuidanceEnv(jsbsim_path="/Users/walter/thesis_project/jsbsim", max_episode_time_s=60 * 5)


register_env("guidance-v0", env_creator)

current_episode = 0
def my_train_fn(config, reporter):
    # agent.restore('/content/drive/MyDrive/checkpoints/checkpoint_701/checkpoint-701')
    agent = TD3Trainer(config=config, env="guidance-v0")

    for i in range(100):
        result = agent.train()
        if i % 10 == 0:
            checkpoint = agent.save(checkpoint_dir="./data/checkpoints")
            print(pretty_print(result))
            print("checkpoint saved at", checkpoint)
    agent.stop()


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, episode, **kwargs):
        envs = base_env.get_unwrapped()

        env_counter = 0
        for env in envs:
            if hasattr(env, "render"):
                rgb_array: np.array = env.render()
                image: Image = Image.fromarray(rgb_array)
                image.save(f'./data/images/episode_{episode.episode_id}_env_{env_counter}.png')

            env_counter += 1

if __name__ == "__main__":
    ray.init()

    config = {
        "lr": 0.01, # tune.grid_search([0.01, 0.001, 0.0001]),
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "tf",
        "eager_tracing": False,
        "monitor": True,
        "callbacks": MyCallbacks,
    }
    resources = TD3Trainer.default_resource_request(config).to_json()
    tune.run(my_train_fn, resources_per_trial=resources, config=config)