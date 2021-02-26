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


def in_seconds(minutes: int) -> int:
    return minutes * 60


def env_creator(env_config=None):
    return NormalizeStateEnv(GuidanceEnv(
        jsbsim_path="/Users/walter/thesis_project/jsbsim",
        max_distance_km=4,
        max_target_distance_km=2,
        max_episode_time_s=60 * 5))

register_env("guidance-v0", env_creator)

current_episode = 0
def my_train_fn(config, reporter):
    # agent.restore('/content/drive/MyDrive/checkpoints/checkpoint_701/checkpoint-701')
    agent = TD3Trainer(config=config, env="guidance-v0")

    for i in range(2):
        result = agent.train()
        if i % 1 == 0:
            checkpoint = agent.save(checkpoint_dir="./data/checkpoints")
            # print(pretty_print(result))
            print("checkpoint saved at", checkpoint)
    agent.stop()


class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, episode, **kwargs):
        envs = base_env.get_unwrapped()
        env_counter = 0
        info = episode.last_info_for()

        for env in envs:
            if hasattr(env, "render"):
                rgb_array: np.array = env.render()
                image: Image = Image.fromarray(rgb_array)

                if info["is_aircraft_out_of_bounds"]:
                    image.save(f'./data/images/out_of_bounds/episode_{episode.episode_id}_env_{env_counter}.png')

                if info["is_aircraft_at_target"]:
                    image.save(f'./data/images/reached_target/episode_{episode.episode_id}_env_{env_counter}.png')

                if not info["is_aircraft_at_target"] and not info["is_aircraft_out_of_bounds"]:
                    image.save(f'./data/images/other/episode_{episode.episode_id}_env_{env_counter}.png')

            env_counter += 1
        #def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        if "num_aircraft_out_of_bounds_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_out_of_bounds_metric"] = 0
        if "num_aircraft_at_target_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_at_target_metric"] = 0
        if "num_aircraft_not_at_target_neither_out_of_bounds_metric" not in episode.custom_metrics:
            episode.custom_metrics["num_aircraft_not_at_target_neither_out_of_bounds_metric"] = 0

        if info["is_aircraft_out_of_bounds"]:
            episode.custom_metrics["num_aircraft_out_of_bounds_metric"] += 1
            print(info["is_aircraft_out_of_bounds"])

        if info["is_aircraft_at_target"]:
            episode.custom_metrics["num_aircraft_at_target_metric"] += 1
            print(info["is_aircraft_at_target"])

        if not info["is_aircraft_at_target"] and not info["is_aircraft_out_of_bounds"]:
            episode.custom_metrics["num_aircraft_not_at_target_neither_out_of_bounds_metric"] += 1

        # print("num_aircraft_out_of_bounds_metric", episode.custom_metrics["num_aircraft_out_of_bounds_metric"])
        # print("num_aircraft_at_target_metric", episode.custom_metrics["num_aircraft_at_target_metric"])
        # print("num_aircraft_not_at_target_neither_out_of_bounds_metric", episode.custom_metrics["num_aircraft_not_at_target_neither_out_of_bounds_metric"])


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
        "callbacks": CustomCallbacks,
    }
    resources = TD3Trainer.default_resource_request(config).to_json()
    tune.run(my_train_fn, resources_per_trial=resources, config=config)