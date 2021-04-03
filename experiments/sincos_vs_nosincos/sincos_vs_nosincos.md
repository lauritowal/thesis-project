# Punish for goal difference
Experiment Date: 03 April 2021, Time: 15:23
## What is the experiment about
Comparison of wrapping cos() and sin() around every angle in observation and two values for x,y in action space to
no wrapping of cos() and sin() and setting the angle directly in the observation space and
directly making the agent choose an angle from 0-360° (continous)


# corresponding branch
experiment_sin_cos

experiment_no_sin_cos

commit:

# Step (for experiment_sin_cos only)

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

        action_target_heading_deg = heading % 360`
        ...
``

# Observation function (for experiment_sin_cos only)
```
    def _get_observation(self) -> np.array:
        aircraft_position = self.sim.get_aircraft_geo_position()

        bearing_to_target_deg = aircraft_position.true_bearing_deg_to(destination=self.target_position)
        current_distance_to_target_m = aircraft_position.distance_haversine_km(self.target_position) * 1000
        max_possible_travelled_distance_m = self.aircraft.get_max_distance_m(episode_time_s=self.max_episode_time_s)

        bearing_to_target_norm = np.interp(math.radians(bearing_to_target_deg), [0, 2*np.pi], [-1, 1])
        current_distance_to_target_norm = np.interp(current_distance_to_target_m, [0, max_possible_travelled_distance_m], [-1, 1])
        runway_angle_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        runway_angle_error_norm = np.interp(runway_angle_error_deg, [-180, 180], [-1, 1])


        return np.array([
            math.sin(math.radians(bearing_to_target_deg)),
            math.cos(math.radians(bearing_to_target_deg)),
            current_distance_to_target_norm,
            math.sin(math.radians(runway_angle_error_deg)),
            math.cos(math.radians(runway_angle_error_deg))
        ], dtype=np.float32)
```

# Reward function (for experiment_sin_cos only)
```
    def _reward(self):
        reward_bounds = 0
        if self.sim.is_aircraft_out_of_bounds(self.max_distance_km):
            reward_bounds = -2

        reward_target_reached = 0
        reward_distance = 0

        if self.sim.is_aircraft_at_target(min_distance_to_target=self.target_radius_km,
                                          target_position=self.target_position):
            runway_angle_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
            if abs(runway_angle_error_deg) < self.runway_angle_threshold_deg:
                print(">>>>>>>>>>>>>>>> error_deg", runway_angle_error_deg)
                reward_target_reached = 1
                print(f"Episode: {self.episode_counter}, "
                      f"runway angle: {self.runway_angle_deg}, "
                      f"heading: {self.sim.get_heading_true_deg()}")
                self.runway_angle_threshold_deg -= 1
                if self.runway_angle_threshold_deg < self.min_runway_angle_threshold_deg:
                    self.runway_angle_threshold_deg = self.min_runway_angle_threshold_deg
                print("new self.runway_angle_threshold_deg", self.runway_angle_threshold_deg)

            else:
                reward_target_reached = - np.interp(abs(runway_angle_error_deg), [0, 180], [0, 1])
        else:
            aircraft_position = self.sim.get_aircraft_geo_position()
            current_distance_to_target_km = aircraft_position.distance_haversine_km(self.target_position)
            reward_distance = -current_distance_to_target_km / 100

        reward = reward_target_reached  + reward_bounds + reward_distance
        assert not math.isnan(reward)

        return reward
```

# Algorithm
## Used Algorithm
TD3
## Used Framework
Rllib
## Algorithm Hyperparams
```
"lr": 0.0001
 Rest to default...
```

# Results
Number of episodes: 1260
Number of steps:


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
Using sin, cos and seems to be more stable in the long term for hitting target and increasing rewards.

# Next Steps
Use sin / cos in observation and two values for action space for now