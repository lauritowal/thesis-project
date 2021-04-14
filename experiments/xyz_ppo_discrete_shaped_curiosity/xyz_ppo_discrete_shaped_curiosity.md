# Punish for goal difference
Experiment Date: 14 April 2021, Time: 14:44
## What is the experiment about
Using x,y,z only in observation to get aircraft to target. 

# corresponding branch
experiment_ppo_discrete_shaped_curiosity

# step
```
    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        heading = 0
        if self.continuous:
            # for continuous action space: invert normalizaation and unpack action
            # action = utils.invert_normalization(x_normalized=action[0], min_x=0.0, max_x=360.0, a=-1, b=1)
            x = action[0]
            y = action[1]
            heading = math.degrees(math.atan2(y, x))
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

# Reward function (for experiment_sin_cos only)
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
PPO + Curiosity 
## Used Framework
Rllib
## Algorithm Hyperparams
```
"lr": 0.0001
 Rest to default...
```

# Results
Number of episodes: 720
Number of steps:


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
Does take way more time to learn... Does not seems worth the effort for xyz.
Could maybe be useful if angle error is added to observation...

# Next Steps
