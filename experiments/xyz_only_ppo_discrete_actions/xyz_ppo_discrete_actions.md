# Punish for goal difference
Experiment Date: 06 April 2021, Time: 19:58
## What is the experiment about
Using x,y,z only in observation to get aircraft to target. 

# corresponding branch
experiment_xyz_ppo_discrete_actions

#action space 
action_space: gym.Space = gym.spaces.Discrete(36)

# step
```
    def step(self, action: np.ndarray):
        action_target_heading_deg = (action + 1) * 10 # action from

        print("action", action)

        self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
                                                                       pitch_angle_current=self.sim[prp.pitch_rad],
                                                                       pitch_angle_rate_current=self.sim[prp.q_radps])
            ...
```

# Observation function (for experiment_sin_cos only)
```
    def _get_observation(self) -> np.array:
        aircraft_position = self.sim.get_aircraft_geo_position()

        wgs84 = nv.FrameE(name='WGS84')
        pointA = wgs84.GeoPoint(latitude=aircraft_position.lat_deg,
                                longitude=aircraft_position.long_deg,
                                z=self.sim[prp.altitude_sl_ft], degrees=True)
        pointB = wgs84.GeoPoint(latitude=self.target_position.lat_deg,
                                longitude=self.target_position.long_deg,
                                z=self.sim[prp.altitude_sl_ft], degrees=True)
        p_AB_N = pointA.delta_to(pointB)
        x, y, z = p_AB_N.pvector.ravel() / 1000

        return np.array([
            x,
            y,
            z
        ], dtype=np.float32)
```

# Reward function 
```
    def _reward(self):
        reward_bounds = 0
        if self.sim.is_aircraft_out_of_bounds(self.max_distance_km):
            reward_bounds = -2

        reward_target_reached = 0

        if self.sim.is_aircraft_at_target(min_distance_to_target=self.target_radius_km,
                                          target_position=self.target_position):
            return 1
        else:
            # reward shaping...
            aircraft_position = self.sim.get_aircraft_geo_position()
            current_distance_to_target_km = aircraft_position.distance_haversine_km(self.target_position)
            reward_distance = -current_distance_to_target_km / 100

        reward = reward_target_reached + reward_bounds + reward_distance
        assert not math.isnan(reward)

        return reward
```

# Algorithm
## Used Algorithm
PPO
## Used Framework
Rllib
## Algorithm Hyperparams
```
default...
```

# Results
Number of episodes: 300-400
Number of steps:


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
Does converge at around 250 episodes (quite fast)
# Next Steps
