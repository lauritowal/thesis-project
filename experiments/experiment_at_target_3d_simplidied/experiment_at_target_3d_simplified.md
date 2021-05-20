# Punish for goal difference
Experiment Date: 12 May 2021, Time: 13:59
## What is the experiment about
Add altitude pid controller and action for altitude to give agent possibility 
to descent more rapidly if needed (altitude_delta_ft)

Also only stop episode when target is reached in 3D space. 
This means also the height is important now.

The agent should learn to get to the target by selecting the correct position 
but also the correct height to descent more rapidly if needed.

# corresponding branch
experiment_at_target_3d

See checkpoint __ in same folder

# step
```
    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        heading_deg = 0
        altitude_delta_ft = 0
        if self.continuous:
            # for continuous action space: invert normalizaation and unpack action
            # action = utils.invert_normalization(x_normalized=action[0], min_x=0.0, max_x=360.0, a=-1, b=1)
            x = action[0]
            y = action[1]
            heading_deg = math.degrees(math.atan2(y, x))
            altitude_delta_ft = np.interp(abs(action[2]), [-1, 1], [-100, 100])

        print("altitude_delta_ft:", altitude_delta_ft)

        action_target_heading_deg = heading_deg % 360
        # self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
        #                                                                pitch_angle_current=self.sim[prp.pitch_rad],
        #                                                                pitch_angle_rate_current=self.sim[prp.q_radps])


        ground_speed = np.sqrt(np.square(self.sim[prp.v_north_fps]) + np.square(self.sim[prp.v_east_fps]))

        # replace with flight_path_angle_hold
        self.sim[prp.elevator_cmd] = self.pid_controller.altitude_hold(altitude_reference_ft=self.sim[prp.altitude_sl_ft] + altitude_delta_ft,
                                                                     altitude_ft=self.sim[prp.altitude_sl_ft],
                                                                     ground_speed=ground_speed,
                                                                     pitch_rad=self.sim[prp.pitch_rad],
                                                                     alpha_rad=self.sim[prp.alpha_rad],
                                                                     roll_rad=self.sim[prp.roll_rad],
                                                                     q_radps=self.sim[prp.q_radps],
                                                                     r_radps=self.sim[prp.r_radps])


        # self.sim[prp.elevator_cmd] = self.pid_controller.flight_path_angle_hold(gamma_reference_rad=math.radians(-5),
        #                                                                       pitch_rad=self.sim[prp.pitch_rad],
        #                                                                       alpha_rad=self.sim[prp.alpha_rad],
        #                                                                       q_radps=self.sim[prp.q_radps],
        #                                                                       roll_rad=self.sim[prp.roll_rad],
        #                                                                       r_radps=self.sim[prp.r_radps])

        self.sim[prp.aileron_cmd] = self.pid_controller.heading_hold(
            heading_reference_deg=action_target_heading_deg,
            heading_current_deg=self.sim.get_heading_true_deg(),
            roll_angle_current_rad=self.sim[prp.roll_rad],
            roll_angle_rate=self.sim[prp.p_radps],
            true_air_speed=self.sim.get_true_air_speed()
        )
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

        altitude_ft = self.sim[prp.altitude_sl_ft]
        altitude_rate_fps = self.sim[prp.altitude_rate_fps]

        in_area = self._in_area()
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position,
                                                             self.localizer_perpendicular_position)

        distance_to_target = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        return np.array([
            in_area,
            cross_track_error,
            altitude_ft / GuidanceEnv.MAX_HEIGHT_FT,
            altitude_rate_fps / GuidanceEnv.MAX_HEIGHT_FT,
            distance_to_target,
            true_airspeed / 1000,
            turn_rate,
            diff.x,
            diff.y,
            diff.z,
            math.sin(math.radians(self.sim.get_heading_true_deg())),
            math.cos(math.radians(self.sim.get_heading_true_deg())),
            math.sin(math.radians(runway_heading_error_deg)),
            math.cos(math.radians(runway_heading_error_deg))
        ], dtype=np.float32)
```

# Reward function (for experiment_sin_cos only)
```
    def _reward(self):
        if self.sim.is_aircraft_altitude_to_low(GuidanceEnv.CRASH_HEIGHT_FT):
            return -10

        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.direction_to_target_deg(self.aircraft_cartesian_position()) - self.runway_angle_deg) % 360

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_heading_error_deg) < self.runway_angle_threshold_deg

        aircraft_position = self.aircraft_cartesian_position()
        diff_position = self.target_position - aircraft_position

        # Really needed or -10 above enough?
        # reward_altitude = -1 + self.sim[prp.altitude_sl_ft] / GuidanceEnv.MAX_HEIGHT_FT

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and is_heading_correct and abs(diff_position.z) <= GuidanceEnv.HEIGHT_THRESHOLD_M / 1000:
            heading_bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + heading_bonus - abs(diff_position.z)

            # reward for height
            print("diff_position.z", diff_position.z)
            print("GuidanceEnv.HEIGHT_THRESHOLD_M", GuidanceEnv.HEIGHT_THRESHOLD_M / 1000)

            return reward

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                       target_position=self.target_position,
                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and not (90 <= relative_bearing_to_aircraft_deg <= 270):
            return -10

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                       target_position=self.target_position,
                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and abs(diff_position.z) > GuidanceEnv.HEIGHT_THRESHOLD_M / 1000:
            return - abs(diff_position.z) * 4

        in_area = self._in_area()

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


        reward_cross_shaped = - abs(cross_track_error) / self.max_distance_km
        reward_altitude_shaped = - abs(diff_position.z) / 10
        ## TODO: Test this out... I think it does not work... maybe though positive rewards for altitude do help since it should help to make aircraft to longer
        # alpha = 0.7
        # reward_shaped = (1-alpha) * reward_cross_shaped + alpha * reward_altitude_shaped
        #
        # print("reward_altitude_shaped", reward_altitude_shaped)
        # print("reward_cross_shaped", reward_cross_shaped)

        reward_shaped = reward_cross_shaped + 3*reward_altitude_shaped

        # print("reward_cross_shaped", reward_cross_shaped)
        # print("reward_altitude_shaped", 3*reward_altitude_shaped)
        # print("reward_shaped", reward_shaped)

        reward_sparse = reward_cross + reward_heading + area_2_penalty

        # a= 1
        # s= 10 / 1000
        # reward = a * math.exp(-(cross_track_error ** 2) / 2*s)


        return reward_shaped + reward_sparse
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
        "seed": SEED
    }
# Results
Number of episodes: 
Number of steps:


# Seed
"seed": 5


### Example images in the end of training (10)

### Description
- Seems to have learned to go to the correct area fast
- Altitude loss is to high. Agent seems to randomly select deltas...

- Does not get to the target, but to the correct direction more or less 
and circles around in the direction of the target (this is only valid when allowing to select altitude_ft directly as action instead of altitude_delta_ft)

### Graph for all seeds

## Conclusion Description

# Next Steps
- Do implement vertical cross track error to a 3D line and make 
aircraft reduce this error (vertical and horizontal)
- Draw 3d line into map to better
-instead of altitude controller use flight path controller to regulate the angle of aircraft or even none...
