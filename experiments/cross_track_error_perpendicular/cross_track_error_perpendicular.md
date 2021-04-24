# Punish for goal difference
Experiment Date: 23 April 2021, Time: 22:37
## What is the experiment about
The cross track error to the line, which is perpendicular to the runway, 
is being calculated if the aircraft is behind the runway direction. 
The aircraft should then get to the correct area first before being then 
rewarded for the other cross track error, which is a normal to the runway.

The agent is rewarded if getting closer (diff) to the area, if not punished.

## What could happen?
- The aircraft could end up cruising around the perpendicular line to get points or to avoid punishment 

# corresponding branch
experiment_relative_bearing_cross_track_error
__
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


        # TODO: Replace target with localizer here?
        cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_position,
                                                         self.runway_angle_deg)

        cross_track_error_perpendicular = self._calc_cross_track_error(current_position=aircraft_position,
                                                         target_position=self.localizer_perpendicular_position,
                                                         heading=self.runway_angle_deg + 90)

        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.true_bearing_deg(aircraft_position) - self.runway_angle_deg) % 360
        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            relative_bearing_to_aircraft_deg = 0

        distance_to_target = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        return np.array([
            relative_bearing_to_aircraft_deg,
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
        if self.sim.is_aircraft_out_of_bounds(self.max_distance_km):
            return -10

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_heading_error_deg) < self.runway_angle_threshold_deg
        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and is_heading_correct:
            bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + bonus
            return reward

        aircraft_position = self.aircraft_cartesian_position()

        cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_position, self.runway_angle_deg)

        diff = (abs(self.last_cross_track_error) + abs(cross_track_error)) / 2

        current_distance_km = aircraft_position.distance_to_target(self.target_position)

        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.true_bearing_deg(aircraft_position) - self.runway_angle_deg) % 360

        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            print("area 1")
            reward_heading = 0
            if abs(diff) < 0.1 and current_distance_km < self.last_distance_km[-1]:

                diff_headings = abs(math.radians(utils.reduce_reflex_angle_deg(runway_heading_error_deg - self.last_runway_heading_error_deg[-1])) / math.pi)
                if abs(runway_heading_error_deg) < abs(self.last_runway_heading_error_deg[-1]):
                    reward_heading = diff_headings
                else:
                    reward_heading = -diff_headings

                reward_cross = 1 # TODO: Do the same as with diff_headings... Might be easier for agent to correct like this maybe...
                # This would maybe also remove the need for the additional reward_cross_shaped... las (cross_track error - current cross track error) / max_distance_km...
            else:
                reward_cross = -1

            self.last_distance_km.append(current_distance_km)
            self.last_runway_heading_error_deg.append(runway_heading_error_deg)

            reward_cross_shaped = - abs(cross_track_error) / self.max_distance_km
            reward = (reward_cross + reward_cross_shaped + reward_heading )
            # a= 1
            # s= 10 / 1000
            # reward = a * math.exp(-(cross_track_error ** 2) / 2*s)

            self.last_cross_track_error = cross_track_error
        else:
            reward = - abs(relative_bearing_to_aircraft_deg) / 180

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
Number of episodes: 
Number of steps:


## Conclusion Description

