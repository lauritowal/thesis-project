# Punish for goal difference
Experiment Date: 21 April 2021, Time: 17:25
## What is the experiment about
Punish / Reward for heading angle difference to runway only

# corresponding branch
experiment_keep_constant_angle


# Observation function (for experiment_sin_cos only)
```
    def _get_observation(self) -> np.array:
        aircraft_position = self.aircraft_cartesian_position()

        diff = self.target_position - aircraft_position

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        # true_airspeed = self.sim.get_true_air_speed()
        # # yaw_rate = self.sim[prp.r_radps]
        # turn_rate = self.sim.get_turn_rate()


        cross_track_error = self._calc_cross_track_error(current_position=aircraft_position,
                                                         target_position=self.target_position)


        distance_to_target = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        return np.array([
            # runway_heading_error,
            # true_airspeed / 1000,
            # turn_rate,
            math.sin(math.radians(self.sim.get_heading_true_deg())),
            math.cos(math.radians(self.sim.get_heading_true_deg())),
            math.sin(math.radians(runway_heading_error_deg)),
            math.cos(math.radians(runway_heading_error_deg))
        ], dtype=np.float32)
```

# Reward function 
```
    def _reward(self):
        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_heading_error_deg) < self.runway_angle_threshold_deg

        if is_heading_correct:
            reward = 1
        else:
            reward = -1
        reward = - math.radians(abs(runway_heading_error_deg) / 180)

        return reward
```

# Algorithm
## Used Algorithm
TD3
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
- Learns to find the correct angle fast... 
- I've noticed that the localizer position is calculated wrong though 
# Next Steps
