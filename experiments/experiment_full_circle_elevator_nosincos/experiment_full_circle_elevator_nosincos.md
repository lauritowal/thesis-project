# Elevator 7 [VEERY GOOD!]
Experiment Date: 27 June 2021, Time: 10:53
## What is the experiment about

Removed the sin() / cos() from all angles in action and observation

# corresponding branch
experiment_full_circle_elevator_nosincos

# tensorboard / ray results: 
See google colab: TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42 (9901 checkpoint)
TD3_guidance-continuous-v0_2021-06-28_11-18-46q1pl2fw5 (18102 checkpoint)

# generate target position 
    def _generate_random_target_position(self) -> (CartesianPosition, float):
        start_distance = 600 / 1000

        def random_sign():
            if self.np_random.random() < 0.5:
                return 1
            return -1

        x = self.np_random.uniform(0, self.max_target_distance_km) * random_sign()
        y = self.np_random.uniform(start_distance, self.max_target_distance_km) * random_sign()
        z = self.np_random.uniform(0.2, (self.sim[prp.initial_altitude_ft] / 3281) / 2) + GuidanceEnv.MIN_HEIGHT_FOR_FLARE_M / 1000

        return CartesianPosition(x, y, z, heading=self.runway_angle_deg, offset=self.offset)

# step
```
    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        heading_deg = 0
        _delta_ft = 0
        if self.continuous:
            heading_deg = np.interp(action[0], [-1, 1], [0, 360])

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

        return np.array([
            in_area,
            cross_track_error,
            vertical_track_error,
            abs(rest_height_ft),
            altitude_rate_fps / GuidanceEnv.MAX_HEIGHT_FT,
            distance_to_target_km,
            true_airspeed / 1000,
            turn_rate,
            diff.x,
            diff.y,
            diff.z,
            np.interp(self.sim.get_heading_true_deg(), [0, 360], [-1, 1]),
            np.interp(runway_heading_error_deg, [0, 360], [-1, 1]),
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
        "num_gpus": 1,
        # "num_workers": 1,
        "num_envs_per_worker": 3,
        "seed": SEED,
        "evaluation_config": {
            "explore": False
        },
        "evaluation_num_workers": 1,
        "env_config": {
            "jsbsim_path": JSBSIM_PATH_DRIVE,
            "flightgear_path": "",
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
            "evaluation": False,
            "seed": SEED,
        }
}
```


# Results
Number of episodes: 10.000
Number of steps:  around 10 M


# Seed
"seed": 4 (Colab)
TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42
TD3_guidance-continuous-v0_2021-06-28_11-18-46q1pl2fw5 (18102 checkpoint)

### Example images in the end of training (10)

Seems to take some more time to learn to separate between 0 and 360, 
but then it does catch up quite good.

Performance at 10.000 steps is not that good as experiment_full_circle_elevator_7 
However, it seems numbers of headings and targets is heigher for experiment_full_circle_lelevator_nosincos in evaluation.
But maybe just the counting is bad in newest evaluation... check again.

### Description

checkpoint 3501 performs pretty good. See evaluation


# Evaluation:

SEED=3

TRAINING SEED=4 (Colab):
- TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42
- checkpoint 9901

TD3_guidance-continuous-v0_2021-06-28_11-18-46q1pl2fw5 (18102 checkpoint)

EVALUATION SEED=1
```
std_reward 328.05389098358137
mean_reward -83.99268834175919
at target 62
on tracks 46
headings_sum 46
others_sum 38
bounds_sum 0
num total episodes 100
distances_global_mean 0.31038647312657897
distances_on_track_mean 0.09441663347509985
runway_angle_errors (all) 14.703203343682771
runway_angle_errors (on track) 4.286847552479082
success total 67
success 0.67
```
EVALUATION SEED=2
```
std_reward 443.60062957423554
mean_reward -143.94783509720722
at target 64
on tracks 61
headings_sum 71
others_sum 34
bounds_sum 2
num total episodes 100
distances_global_mean 0.39986210877495215
distances_on_track_mean 0.09498260776015166
runway_angle_errors (all) 18.47888681963449
runway_angle_errors (on track) 4.726107289241195
success total 71
success 0.71
```
EVALUATION SEED=3
```
std_reward 409.12386224115266
mean_reward -139.18037076943298
at target 62
on tracks 54
headings_sum 70
others_sum 38
bounds_sum 0
num total episodes 100
distances_global_mean 0.27659839676967996
distances_on_track_mean 0.11055000964056283
runway_angle_errors (all) 12.994131153595992
runway_angle_errors (on track) 4.3071023953631355
success total 70
success 0.7
```

TRAINING SEED=4 (Colab):
- TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42
- checkpoint 18102 (restored and continued from checkpoint 9901)

EVALUATION SEED=1
```
std_reward 330.55382959186437
mean_reward -91.8633373943261
at target 77
on tracks 82
headings_sum 85
others_sum 23
bounds_sum 0
num total episodes 100
distances_global_mean 0.15875109134680665
distances_on_track_mean 0.10485365630921112
runway_angle_errors (all) 15.578021766303078
runway_angle_errors (on track) 11.965882926511382
success total 85
success 0.85
```

EVALUATION SEED=2
```
std_reward 404.233319437487
mean_reward -136.75364777148047
at target 72
on tracks 76
headings_sum 81
others_sum 28
bounds_sum 0
num total episodes 100
distances_global_mean 0.29352448366489386
distances_on_track_mean 0.11397173925187143
runway_angle_errors (all) 18.189484895818964
runway_angle_errors (on track) 12.699177564392986
success total 81
success 0.81
```
EVALUATION SEED=3
```
std_reward 373.28571073930834
mean_reward -173.1808436157436
at target 70
on tracks 74
headings_sum 78
others_sum 30
bounds_sum 0
num total episodes 100
distances_global_mean 0.2151672474840204
distances_on_track_mean 0.1246633611884744
runway_angle_errors (all) 17.671178427716512
runway_angle_errors (on track) 12.70653390236578
success total 78
success 0.78
```


# Next Steps:
- Train even longer --> there is still growth potential!
- Train other two seeds: 3, 7