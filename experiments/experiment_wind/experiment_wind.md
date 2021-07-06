# Wind [VEERY GOOD!]
Experiment Date: 18 June 2021, Time: 12:28
## What is the experiment about

Add random constant wind from east and west to environment

# corresponding branch
experiment_wind

# tensorboard / ray results: 
See google colab: 
TD3_guidance-continuous-v0_2021-06-17_16-29-02vu2kd674

# reset
    def reset(self):
        initial_conditions = self._get_initial_conditions()

        if self.sim:
            self.sim.reinitialise(init_conditions=initial_conditions)
        else:
            self.sim = self._init_new_sim(self.JSBSIM_DT_HZ, self.aircraft, initial_conditions)

        self.steps_left = self.episode_steps

        ###### WIND
        self.sim[prp.wind_east_fps] = 0 # self.np_random.uniform(-70, 70)
        self.sim[prp.wind_north_fps] = 0 # self.np_random.uniform(-70, 70) # leave north wind at 0 for simplification for now

        # TODO: find more elegant solution...
        if self.phase == 0:
            self.spawn_target_distance_km = 0.5
            # self.runway_angle_deg = self.np_random.uniform(-20, 20) % 360
            self.sim[prp.wind_east_fps] = 0  # self.np_random.uniform(-70, 70)
        elif self.phase == 1:
            self.spawn_target_distance_km = 1
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-10, 10)
            # self.runway_angle_deg = self.np_random.uniform(-45, 45) % 360
        elif self.phase == 2:
            self.spawn_target_distance_km = 1.5
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-20, 20)
            # self.runway_angle_deg = self.np_random.uniform(-90, 90) % 360
        elif self.phase == 3:
            self.spawn_target_distance_km = 2
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-35, 35)
            # self.runway_angle_deg = self.np_random.uniform(-120, 120) % 360
        elif self.phase == 4:
            self.spawn_target_distance_km = self.max_target_distance_km
            self.sim[prp.wind_east_fps] = 55 * (1 if self.np_random.random() < 0.5 else -1)

# step
```
    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        heading_deg = 0
        _delta_ft = 0
        if self.continuous:
            # for continuous action space: invert normalizaation and unpack action
            # action = utils.invert_normalization(x_normalized=action[0], min_x=0.0, max_x=360.0, a=-1, b=1)
            x = action[0]
            y = action[1]
            heading_deg = math.degrees(math.atan2(y, x))
            # altitude_delta_ft = np.interp(abs(action[2]), [-1, 1], [-100, 100])

        # print("altitude_delta_ft:", altitude_delta_ft)

        action_target_heading_deg = heading_deg % 360
        self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
                                                                       pitch_angle_current=self.sim[prp.pitch_rad],
                                                                       pitch_angle_rate_current=self.sim[prp.q_radps])


        ground_speed = np.sqrt(np.square(self.sim[prp.v_north_fps]) + np.square(self.sim[prp.v_east_fps]))

        # replace with flight_path_angle_hold
        # self.sim[prp.elevator_cmd] = self.pid_controller.altitude_hold(altitude_reference_ft=self.sim[prp.altitude_sl_ft] + altitude_delta_ft,
        #                                                              altitude_ft=self.sim[prp.altitude_sl_ft],
        #                                                              ground_speed=ground_speed,
        #                                                              pitch_rad=self.sim[prp.pitch_rad],
        #                                                              alpha_rad=self.sim[prp.alpha_rad],
        #                                                              roll_rad=self.sim[prp.roll_rad],
        #                                                              q_radps=self.sim[prp.q_radps],
        #                                                              r_radps=self.sim[prp.r_radps])


        # self.sim[prp.elevator_cmd] = self.pid_controller.flight_path_angle_hold(gamma_reference_rad=math.radians(0),
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

        for step in range(self.sim_steps):
            self.sim.run()

        reward = self._reward()
        state = self._get_observation()

        self.rewards.append(reward)
        self.last_state = state
        self.steps_left -= 1

        self.done = self._is_done()

        info = self.get_info(reward=reward)
        self.infos.append(info)

        if self.render_progress_image and self._is_done():
            rgb_array = self.render(mode="rgb_array")
            image: Image = Image.fromarray(rgb_array)
            image.save(f'{self.render_progress_image_path}/episode_{self.episode_counter}_{info["terminal_state"]}.png')
            print("done with episode: ", self.episode_counter)

        self.episode_counter += 1
        return state, reward, self.done, info
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
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)


        distance_to_target_km = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km

        rest_height_ft = (diff.z * 3281) / GuidanceEnv.MAX_HEIGHT_FT # altitude_ft / GuidanceEnv.MAX_HEIGHT_FT

        # wind direction
        # wind speed
        # ground speed
        # drift angle (track angle - heading) needed?

        drift_deg = self.get_drift_deg()
        # print("drift", drift_deg)
        # print("track", self.sim.get_track_angle_deg())
        # print("heading", self.sim.get_heading_true_deg())
        # print("self.sim[prp.total_wind_east_fps]", self.sim[prp.total_wind_east_fps])
        # print("self.sim[prp.total_wind_north_fps]", self.sim[prp.total_wind_north_fps])
        # print("self.sim[prp.total_wind_down_fps]", self.sim[prp.total_wind_down_fps])

        return np.array([
            in_area,
            cross_track_error,
            vertical_track_error,

            # wind
            self.sim[prp.total_wind_north_fps],
            self.sim[prp.total_wind_east_fps],
            ## drift angle really needed, if we have cross track error anyways?
            math.sin(math.radians(drift_deg)),
            math.cos(math.radians(drift_deg)),

            abs(rest_height_ft),
            altitude_rate_fps / GuidanceEnv.MAX_HEIGHT_FT,
            distance_to_target_km,
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
        in_area = self._in_area()
        aircraft_position = self.aircraft_cartesian_position()

        if self.sim.is_aircraft_altitude_to_low(self.to_low_height):
            distance_error = aircraft_position.distance_to_target(self.target_position) * 6
            print("is_aircraft_altitude_to_low: distance_error", distance_error)
            return - np.clip(abs(distance_error), 0, 10)

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM): # and is_heading_correct
            heading_bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + heading_bonus

            print("at target, positive reward: ", reward)

            # reward for height
            return reward

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                       target_position=self.target_position,
                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and not in_area:
            return -10

        current_distance_km = aircraft_position.distance_to_target(self.target_position)
        reward_heading = 0
        reward_track = 0
        penalty_area_2 = 0

        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
            vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

            track_error = abs(cross_track_error) + abs(vertical_track_error)

            diff_track = abs(self.last_track_error - track_error)

            diff_headings = abs(math.radians(utils.reduce_reflex_angle_deg(runway_heading_error_deg - self.last_runway_heading_error_deg[-1])) / math.pi)
            if self._is_on_track():
                if abs(runway_heading_error_deg) < abs(self.last_runway_heading_error_deg[-1]):
                    reward_heading = diff_headings
                else:
                    reward_heading = -diff_headings
                reward_track = 1 # Maybe diff_track * 2 instead of 1
            else:
                reward_track = -diff_track * 2

            self.last_distance_km.append(current_distance_km)
            self.last_runway_heading_error_deg.append(runway_heading_error_deg)
            self.last_track_error = track_error
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
            self.last_track_error_perpendicular = cross_track_error
            track_error = cross_track_error
            penalty_area_2 = -2

        # Maybe replace exp again with linear
        clipped = np.clip(np.exp(abs(track_error)), math.exp(0), math.exp(self.max_distance_km))
        reward_track_shaped = - np.interp(clipped,
                                          [math.exp(0), math.exp(self.max_distance_km)],
                                          [0, 1])

        # reward_altitude_shaped = - abs(diff_position.z) / 10

        reward_shaped = reward_track_shaped # + reward_altitude_shaped
        reward_sparse = reward_track + penalty_area_2 + reward_heading

        return reward_shaped + reward_sparse
```

# Algorithm
## Used Algorithm
TD3
## Used Framework
Rllib
## Algorithm Hyperparams
```
    custom_config = {
        "lr": 0.0001, # tune.grid_search([0.01, 0.001, 0.0001]),
        "framework": "torch",
        "callbacks": CustomCallbacks,
        "log_level": "WARN",
        "evaluation_interval": 20,
        "evaluation_num_episodes": 10,
        "num_gpus": 0,
        "num_workers": 1,
        "num_envs_per_worker": 3,
        "seed": SEED,
        "env_config": {
            "jsbsim_path": "/Users/walter/thesis_project/jsbsim",
            "flightgear_path": "/Users/walter/FlightGear.app/Contents/MacOS/",
            "aircraft": cessna172P,
            "agent_interaction_freq": 5,
            "target_radius": 100 / 1000,
            "max_distance_km": 4,
            "max_target_distance_km": 2,
            "max_episode_time_s": 60 * 5,
            "phase": 0,
            "render_progress_image": False,
            "render_progress_image_path": './data',
            "offset": 0,
            "seed": SEED,
            "evaluation": False,
        },
        "evaluation_config": {
            "explore": False
        },
        "evaluation_num_workers": 1,
    }
```


# Results
Number of episodes: 10.000
Number of steps:  around 10 M


# Seed
"seed": 4 (Colab)
TD3_guidance-continuous-v0_2021-06-17_16-29-02vu2kd674 (around 5600 episodes only.)

### Example images in the end of training (10)

### Description

checkpoint 5301 performs pretty good. See evaluation

Removing the penalty for height seems to make the result better.

# Evaluation:

TRAINING SEED=4 (Colab):
- checkpoint 5301

EVALUATION SEED=1
```
std_reward 359.01656571404544
mean_reward -75.07325537861027
at target 26
on tracks 61
headings_sum 29
others_sum 55
bounds_sum 19
num total episodes 100
distances 1.5622825706118264
runway_angle_errors (all) 54.78740267874805
success total 64
success 0.64
```
EVALUATION SEED=2
```
std_reward 425.4531441424144
mean_reward -175.83711759258642
at target 19
on tracks 47
headings_sum 13
others_sum 56
bounds_sum 25
num total episodes 100
distances 2.1910108666894574
runway_angle_errors (all) 68.21167302944431
success total 51
success 0.51
```
EVALUATION SEED=3
```
std_reward 430.60648842936126
mean_reward -150.7913526166676
at target 28
on tracks 57
headings_sum 22
others_sum 58
bounds_sum 14
num total episodes 100
distances 1.5115976146341656
runway_angle_errors (all) 56.39218103547068
success total 60
success 0.6
```

Avarage: 0.58

# Next Steps:
- Adapt range for wind to -100 to 100