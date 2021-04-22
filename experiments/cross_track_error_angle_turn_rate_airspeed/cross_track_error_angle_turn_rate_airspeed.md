# Punish for goal difference
Experiment Date: 22 April 2021, Time: 18:31
## What is the experiment about
Use track error and reward only if distance to target is reduced..

# corresponding branch
experiment_cross_track_error_angle_turn_rate_airspeed

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
        aircraft_position = self.aircraft_cartesian_position()

        diff = self.target_position - aircraft_position

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        true_airspeed = self.sim.get_true_air_speed()
        # # yaw_rate = self.sim[prp.r_radps]
        turn_rate = self.sim.get_turn_rate()


        cross_track_error = self._calc_cross_track_error(current_position=aircraft_position,
                                                         target_position=self.target_position)


        distance_to_target = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        return np.array([
            cross_track_error,
            distance_to_target,
            true_airspeed / 1000,
            turn_rate,
            diff.x,
            diff.y,
            math.sin(math.radians(self.sim.get_heading_true_deg())),
            math.cos(math.radians(self.sim.get_heading_true_deg())),
            math.sin(math.radians(runway_heading_error_deg)),
            math.cos(math.radians(runway_heading_error_deg))
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
Number of episodes: 1800
Number of steps: 890k


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
Reaches target very often and heading at around 60-70% of the times 
threshold is 10 deg.
runway heading error mean is at around 8-10° 
which even better then the version without considering rewarding the correct heading!

# Next Steps
