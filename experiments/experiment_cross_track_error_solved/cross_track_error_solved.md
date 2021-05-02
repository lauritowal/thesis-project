# Punish for goal difference
Experiment Date: 02 May 2021, Time: 12:08
## What is the experiment about
Calc cross track error to the line perpendincular to the localizer in front of the runway to get the aircraft to the correct area. 
Afterwards calculate the cross track error to the norm of the runway.

# corresponding branch
experiment_cross_track_error_210429

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


        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.true_bearing_deg(self.aircraft_cartesian_position()) - self.runway_angle_deg) % 360
        in_area = False
        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            in_area = True

        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position,
                                                             self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position,
                                                             self.localizer_perpendicular_position)

        distance_to_target = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        return np.array([
            in_area,
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
        # if self.sim.is_aircraft_out_of_bounds(self.max_distance_km):
        #     return -10
        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.true_bearing_deg(self.aircraft_cartesian_position()) - self.runway_angle_deg) % 360

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_heading_error_deg) < self.runway_angle_threshold_deg
        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and is_heading_correct:
            bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + bonus
            return reward

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                       target_position=self.target_position,
                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and not (90 <= relative_bearing_to_aircraft_deg <= 270):
            return -10

        aircraft_position = self.aircraft_cartesian_position()

        in_area = False
        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            in_area = True

        current_distance_km = aircraft_position.distance_to_target(self.target_position)
        reward_heading = 0
        reward_cross = 0
        area_2_penalty = 0
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
            cross_track_medium_error = (abs(self.last_cross_track_error) + abs(cross_track_error)) / 2
            diff_cross = abs(self.last_cross_track_error - cross_track_error)

            diff_headings = abs(math.radians(utils.reduce_reflex_angle_deg(runway_heading_error_deg - self.last_runway_heading_error_deg[-1])) / math.pi)
            if abs(cross_track_medium_error) < 0.1 and current_distance_km < self.last_distance_km[-1] and abs(runway_heading_error_deg) < 90:
                if abs(runway_heading_error_deg) < abs(self.last_runway_heading_error_deg[-1]):
                    reward_heading = diff_headings
                else:
                    reward_heading = -diff_headings
                reward_cross = 1
            else:
                reward_cross = -diff_cross * 2

            self.last_distance_km.append(current_distance_km)
            self.last_runway_heading_error_deg.append(runway_heading_error_deg)
            self.last_cross_track_error = cross_track_error
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
            # cross_track_medium_error = (abs(self.last_cross_track_error_perpendicular) + abs(cross_track_error)) / 2
            self.last_cross_track_error_perpendicular = cross_track_error
            area_2_penalty = -2
```

# Algorithm
## Used Algorithm
TD3
## Used Framework
Rllib
## Algorithm Hyperparams
    custom_config = {
        "lr": 0.0001, # tune.grid_search([0.01, 0.001, 0.0001]),
        "num_gpus": 0,
        "framework": "torch",
        "callbacks": CustomCallbacks,
        "log_level": "WARN",
        "evaluation_interval": 20,
        "evaluation_num_episodes": 10,
        "num_workers": 0,
        "num_envs_per_worker": 3,
        "seed": 1
    }
# Results
Number of episodes: 1587
Number of steps:


# Seed
"seed": 1


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
max median 19° runway angle error!!

# Next Steps
