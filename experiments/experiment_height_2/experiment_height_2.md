# Height
Experiment Date: 24 June 2021, Time: 14:23
## What is the experiment about

Set GuidanceEnv.MAX_HEIGHT_FT to 5000 
--> Try to see if agent learns to get to the target also when starting heigher...

Set max_distance_km back to 4

# corresponding branch
experiment_height_2

# tensorboard / ray results: 
See google colab: TD3_guidance-continuous-v0_2021-06-23_07-08-28k4ozsxw4

# initial conditions
```
    MAX_HEIGHT_FT = 5000

    def _get_initial_conditions(self):
        return {
            # TODO: better min back to 100-500? Seems better for initial training...
            prp.initial_altitude_ft: self.np_random.uniform(GuidanceEnv.MIN_HEIGHT_FT, GuidanceEnv.MAX_HEIGHT_FT),
            prp.initial_terrain_altitude_ft: 0.00000001,
            prp.initial_longitude_geoc_deg: self.np_random.uniform(-160, 160), # decreased range to avoid problems close to eqautor at -180 / 180
            prp.initial_latitude_geod_deg: self.np_random.uniform(-70, 70), # decreased range to avoid problems close to poles at -90 / 90
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.np_random.uniform(0, 360),
        }
```
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
                reward_track = 1
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
Number of episodes: 
Number of steps:  around 


# Seed
"seed": 4 (Colab)
TD3_guidance-continuous-v0_2021-06-20_09-30-149abgf385


### Example images in the end of training (10)

### Description
After setting max_distance_km back to 4 it does peform way better!
-Does not to perform that much better with seed 9801 vs 3601

# Evaluation:

SEED=3

TRAINING SEED=4 (Colab):
- TD3_guidance-continuous-v0_2021-06-20_09-30-149abgf385

- checkpoint --> 3601 

EVALUATION SEED=1
```
std_reward 317.74683654029985
mean_reward -113.11669894024448
at target 23
on tracks 42
headings_sum 33
others_sum 54
bounds_sum 23
num total episodes 100
distances 2.4509740846097854
runway_angle_errors (all) 85.47222919906159
success total 42
success 0.42
```
EVALUATION SEED=2
```

```
EVALUATION SEED=3
```
std_reward 380.20641342126856
mean_reward -201.3673690831826
at target 17
on tracks 43
headings_sum 34
others_sum 54
bounds_sum 29
num total episodes 100
distances 2.7326611424808687
runway_angle_errors (all) 87.33101533037534
success total 44
success 0.44
```

- checkpoint --> 9801 

EVALUATION SEED=1
```
std_reward 352.0526172623021
mean_reward -82.62490997281756
at target 25
on tracks 43
headings_sum 28
others_sum 56
bounds_sum 19
num total episodes 100
distances 2.1539273165659027
runway_angle_errors (all) 75.20519697586047
success total 47
success 0.47
```
EVALUATION SEED=2
```
std_reward 427.33837647543004
mean_reward -128.03279639207815
at target 21
on tracks 42
headings_sum 28
others_sum 62
bounds_sum 17
num total episodes 100
distances 2.1160736751761853
runway_angle_errors (all) 71.24110230174088
success total 45
success 0.45
```
EVALUATION SEED=3
```
std_reward 407.77788489236525
mean_reward -156.2633700649247
at target 17
on tracks 31
headings_sum 21
others_sum 67
bounds_sum 16
num total episodes 100
distances 2.0806569145143787
runway_angle_errors (all) 82.5898645234719
success total 34
success 0.34
```





# Next Steps:
