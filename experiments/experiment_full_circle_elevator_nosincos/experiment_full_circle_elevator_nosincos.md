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
TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42 (9901 episodes)
TD3_guidance-continuous-v0_2021-06-28_11-18-46q1pl2fw5 (18102 checkpoint)

"seed": 3
TD3_guidance-continuous-v0_2021-07-24_22-54-2108w_5fiq (around 10M)

"seed": 7
TD3_guidance-continuous-v0_2021-07-23_16-44-36wz9_dn8_ (around 10M)


### Example images in the end of training (10)

Seems to take some more time to learn to separate between 0 and 360, 
but then it does catch up quite good.

Performance at 10.000 steps is not that good as experiment_full_circle_elevator_7 
However, it seems numbers of headings and targets is heigher for experiment_full_circle_lelevator_nosincos in evaluation.
But maybe just the counting is bad in newest evaluation... check again.

### Description

checkpoint 9901 performs pretty good without sin/cos, but still not as good as 9901 with sin/cos.
checkpoint 18102 however reaches around 81 %
Also Heading somes are pretty high without sin/cos


# Evaluation:

- TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42
SEED=3

TRAINING SEED=4 (Colab):
- TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42
- checkpoint 9901


EVALUATION SEED=1
```
std_reward 328.05389098358137
mean_reward -83.99268834175919
at target 62
on tracks 46
altitude_rates_mean -13.084689266955657
headings_sum 57
alphas mean 5.371477430388453
ground_speed mean 112.26379513991773
pitches mean -1.004821427055866
others_sum 38
bounds_sum 0
num total episodes 100
distances_global_mean 0.31038647312657897
tas_mean (fps) 112.37685252178021
distances_on_track_mean 0.0811632641525618
runway_angle_errors (all) 14.703203343682771
runway_angle_errors (on track) 4.923992115715378
all gammas_deg [4.346616958413997, 4.747200936886509, 4.864274591967105, 4.499686492249823, 4.288350851403299, 4.240561454716109, 4.279232178602765, 3.910005365771052, 4.275068944550188, 4.360826233788076, 4.206263785938216, 4.2702608074380395, 4.319861191484524, 4.8110480321267755, 3.9810188204156094, 4.470069868417031, 4.392629960610684, 4.287242748808974, 4.3121822299726045, 4.648214778839296, 4.491119296723182, 4.2930222334391575, 4.296664760738458, 4.375813098619113, 4.597575690038711, 4.884973240553248, 3.8585233111706416, 4.272973525832419, 4.34779716862416, 4.310022162237095, 4.692741740094594, 3.89344704945892, 4.274223475105037, 4.449833844491726, 4.280469276863473, 4.27020546250627, 4.275502707034166, 4.273547137044553, 4.505193409367671, 4.30294534549827, 4.350783578885473, 4.635059226922776, 4.980792399191742, 4.392921023810417, 4.269980531676732, 4.271962486806815, 3.7303544319548068, 4.154105880909126, 4.564028704385908, 4.262192418479447, 4.279583544000786, 4.284195578588116, 4.754172776958308, 4.888153581176847, 4.300323631341861, 4.240615276184929, 4.48385178534817, 4.334975360831515, 3.968966399854248, 4.680433648985902, 4.273047397059583, 4.540539268794383, 4.276140724001658, 4.285188372263649, 4.6961580484756205, 3.9244451764728403, 4.285770802080142]
gammas_mean (on track) 4.366656003332587
success total 67
success 0.67
```
EVALUATION SEED=2
```
SEED 2
##################################
std_reward 443.60062957423554
mean_reward -143.94783509720722
at target 64
on tracks 61
altitude_rates_mean -12.911101783499896
headings_sum 63
alphas mean 5.336404939155262
ground_speed mean 112.33368399929857
pitches mean -1.0261770774745254
others_sum 34
bounds_sum 2
num total episodes 100
distances_global_mean 0.39986210877495215
tas_mean (fps) 112.46557357577935
distances_on_track_mean 0.09498260776015166
runway_angle_errors (all) 18.47888681963449
runway_angle_errors (on track) 4.726107289241195
all gammas_deg [4.273571074999423, 3.666731692803553, 3.571684576044166, 3.958529059388441, 4.365949675192531, 4.276347571413508, 4.27547309752787, 3.998041857582866, 4.3238026821227855, 4.273858880010941, 4.3346651400639225, 4.354042748913983, 3.350319962516424, 4.611593367209803, 4.433259434689395, 4.298915460895637, 4.273262304941627, 4.331203774000052, 4.359445937055824, 4.271041768027725, 4.270129237029254, 4.377939807823707, 4.292487845875871, 4.358597890639445, 4.961824915195167, 4.272717319935429, 4.671977992199624, 4.556229299727368, 4.319580269062336, 4.276720194222865, 4.40632298998902, 4.600802583538775, 4.270720910487197, 5.148228734116326, 4.279963704293078, 4.376681318442383, 4.288334784260022, 3.8345432022216843, 4.270502126638231, 4.435146994733126, 4.587521773575866, 4.434141698852908, 4.526066468668402, 4.2727623537670745, 4.270090143685703, 4.270278374334471, 4.2870111154174575, 4.272820851007887, 4.277835098728979, 3.853521945986753, 4.202597181448171, 4.27233546500756, 4.271530897159992, 4.734067842063284, 4.465602063328542, 4.276816426104694, 4.2888357921219775, 4.314317170499578, 4.278650742476549, 4.431232260942747, 3.9844545951883057, 4.283043031311565, 4.29647413232387, 4.27438875517375, 4.273315214785819, 4.272881874503153, 4.308621934050738, 4.278330270945521, 4.296980811423591, 4.625271188667268, 4.673188519948661]
gammas_mean (on track) 4.310227861680735
success total 71
success 0.71
```
EVALUATION SEED=3
```
std_reward 409.12386224115266
mean_reward -139.18037076943298
at target 62
on tracks 54
altitude_rates_mean -12.799271015274826
headings_sum 63
alphas mean 5.299606832031635
ground_speed mean 112.42212947737903
pitches mean -1.0195654284636044
others_sum 38
bounds_sum 0
num total episodes 100
distances_global_mean 0.27659839676967996
tas_mean (fps) 112.53361894734223
distances_on_track_mean 0.11055000964056283
runway_angle_errors (all) 12.994131153595992
runway_angle_errors (on track) 4.3071023953631355
all gammas_deg [3.5548901792639205, 4.272399330578858, 4.2736108016248435, 4.293098067494948, 4.530635726156927, 4.517944599070777, 4.442380611937305, 4.3730764283343, 4.974917022937287, 4.3099505378086445, 3.6521698567826246, 3.998837622006454, 4.282164078903248, 4.269958514912145, 4.269980273661651, 4.2715141216402515, 4.326577834815015, 4.028468915557216, 4.410368497958019, 4.071466736167873, 4.282003792053712, 4.306734473435687, 4.625556379749358, 4.335886826259201, 4.278234244076712, 3.89736119137268, 4.272399979544626, 4.270401132107965, 4.129741342355486, 4.829225593342858, 4.274073309370498, 4.445542100206018, 3.9831871061280726, 4.271814080872544, 4.2993887450105355, 4.272589778616192, 4.270028666171311, 4.36243639891608, 4.289574962876613, 4.283369219709096, 4.187335957058201, 4.272798961324133, 4.271805794184979, 4.272397859884873, 4.275284474128077, 4.257900906955947, 4.2770499148183525, 4.282682953170721, 3.970484594497711, 4.389661232706164, 4.294236294168421, 4.27250090754521, 4.197696495054075, 4.359451130811883, 4.359305880066277, 4.378129078565296, 4.634168306518704, 4.256142777528688, 4.2709215307567625, 4.272422074247135, 4.274374443941375, 4.27391903462523, 4.183986799045648, 4.286641177766661, 4.2735144162173055, 4.567214737739545, 3.6155910136256555, 4.348438548487294, 4.652333235999768, 4.2705486385345095]
gammas_mean (on track) 4.280041403568032
success total 70
success 0.7
```

TRAINING SEED=4 (Colab):
TD3_guidance-continuous-v0_2021-06-28_11-18-46q1pl2fw5 (18102 checkpoint)

- checkpoint 18102 (restored and continued from checkpoint 9901)

EVALUATION SEED=1
```
#################################
SEED 1
##################################
std_reward 330.55382959186437
mean_reward -91.8633373943261
at target 77
on tracks 82
altitude_rates_mean -14.725117838410402
headings_sum 41
alphas mean 5.295591641829779
ground_speed mean 111.57601323818575
pitches mean -1.0795955051119173
others_sum 23
bounds_sum 0
num total episodes 100
distances_global_mean 0.15875109134680665
tas_mean (fps) 111.2022453924002
distances_on_track_mean 0.10485365630921112
runway_angle_errors (all) 15.578021766303078
runway_angle_errors (on track) 11.965882926511382
all gammas_deg [4.286299236427111, 3.6745036287244797, 4.200331679615169, 3.984960598889722, 3.7820032766880103, 4.629475698916159, 5.2015688297839375, 3.2561365458316445, 4.336583705631936, 4.378139985235496, 4.567922624791201, 4.881000926511754, 4.887248327457702, 4.758437605707778, 5.125079223244781, 5.3149258485134, 3.45339942283699, 4.3502129879704405, 4.230218327855888, 5.15707214425068, 5.307357968429282, 4.282602348561744, 4.163881504694289, 4.227217132542759, 4.046239595097718, 3.7094036285013425, 4.63107373872422, 3.766551270693908, 4.081003468824578, 4.344281596470572, 3.700864524648536, 4.770421550222446, 3.755360505519847, 3.2966246974899023, 3.1956615923592264, 3.507281674283538, 4.275358711203667, 3.499046033161847, 4.078824060217444, 4.127460597256801, 4.2181026546411715, 4.772099701961003, 3.9235297727924694, 4.7383390428154275, 4.004362678033215, 4.494956104195158, 4.753430297188062, 3.400213534171196, 3.482825483910374, 3.5750747418371516, 4.506679591982328, 3.7054298868750033, 3.4250970579348623, 4.887966984712834, 4.3854861245926084, 3.4487274785692734, 3.2343181635654217, 3.502979782357575, 4.199573323435944, 4.530854820143561, 4.493909976441657, 3.3840903357157672, 4.8526111115143875, 3.752892379780499, 4.173030846978177, 4.468154354057127, 4.118506227386169, 4.228622756845821, 4.092025900480112, 3.5243138667620215, 3.7890893413657905, 4.136499272672881, 3.9414352211232986, 4.964062556278254, 4.064566661590952, 4.350964624078358, 4.238691104426774, 4.166763673066866, 3.5859693624881306, 4.60327244883716, 3.680117969596067, 4.4752683234078265, 4.343136785394722, 4.417261252648046, 2.9920811293524188]
gammas_mean (on track) 4.155875547432846
success total 85
success 0.85

```

EVALUATION SEED=2
```
#################################
SEED 2
##################################
std_reward 404.233319437487
mean_reward -136.75364777148047
at target 72
on tracks 76
altitude_rates_mean -14.584096033117877
headings_sum 35
alphas mean 5.309326974232642
ground_speed mean 111.45252950868895
pitches mean -1.0927528130464956
others_sum 28
bounds_sum 0
num total episodes 100
distances_global_mean 0.29352448366489386
tas_mean (fps) 111.14831647986603
distances_on_track_mean 0.11397173925187143
runway_angle_errors (all) 18.189484895818964
runway_angle_errors (on track) 12.699177564392986
all gammas_deg [3.643333290707986, 4.203440540775696, 4.347142773011971, 4.391572464324766, 3.0701576967793063, 4.258294916780497, 4.798395136828226, 4.2695173763989125, 5.33332730978428, 4.043273753468639, 3.836610627950984, 4.0769646461122235, 3.7633732643736613, 4.289077020539936, 5.131216249125381, 3.3245028321840397, 3.739638082505558, 4.225192332866941, 3.7269680796506175, 4.290708181234982, 4.508995818264253, 3.6101896234075523, 4.242971059712558, 3.504269209207573, 4.006053358765035, 4.072912147131862, 3.959347633568159, 5.402459084246341, 4.981944547782349, 4.022573596281828, 3.9372046458428844, 4.89905877828975, 4.237184351872135, 4.448507920360649, 4.245342517409559, 3.3586204246492346, 4.696583965256838, 4.312731304183898, 4.1430379004294275, 4.6127313908089915, 4.313142118479632, 3.5915646421708938, 4.228482891357173, 4.008706746822941, 3.4290650607757938, 4.776706641103794, 3.5245725293624406, 4.362201914795003, 4.203348457886419, 4.743591886896299, 3.8095538182323394, 4.034424071077126, 4.316644797746449, 3.882300880101791, 4.9008963445093805, 3.480132416381607, 4.499541761053632, 4.2517582185599165, 4.043921110300099, 4.208157704913223, 4.297519225516703, 3.9221293384239964, 4.227141679746978, 3.2521745810249225, 4.691342462771436, 3.7234970465740775, 4.1176382701062355, 4.349530636158995, 5.141002267782722, 3.779506412278793, 4.4809228990399586, 4.515832958606192, 3.82213756632674, 5.4365338274097, 4.068732038416109, 4.416528341488011, 4.221507627788217, 3.774363563517008, 4.53232903165068, 4.810221549644331, 4.283359140370092]
gammas_mean (on track) 4.202939386790658
success total 81
success 0.81

```
EVALUATION SEED=3
```
SEED 3
##################################
std_reward 373.28571073930834
mean_reward -173.1808436157436
at target 70
on tracks 74
altitude_rates_mean -14.483201760442766
headings_sum 32
alphas mean 5.422097549612385
ground_speed mean 111.5273238080751
pitches mean -1.1426127620262279
others_sum 30
bounds_sum 0
num total episodes 100
distances_global_mean 0.2151672474840204
tas_mean (fps) 111.38499018235017
distances_on_track_mean 0.1246633611884744
runway_angle_errors (all) 17.671178427716512
runway_angle_errors (on track) 12.70653390236578
all gammas_deg [4.0162244901074935, 4.519105334739042, 3.866069026727214, 3.640748957068903, 4.576027155555005, 4.4218109605905145, 5.058938447590619, 4.298210724041393, 4.196689871125128, 4.152820367137294, 4.859806399696977, 3.8831950381375386, 4.603074329234545, 4.815303681352855, 3.829819086349181, 5.295815898969779, 3.619134834548163, 4.122834715695926, 4.225043097610396, 3.097144885462531, 4.502497670349171, 4.372440919004858, 4.3473680617446355, 4.256889359659924, 3.379426757059857, 5.428617529777584, 4.238792013996137, 5.309109677126613, 3.7513345372554454, 3.7560188215559016, 4.2457631046954125, 4.312068326859571, 4.514237891570721, 4.122249428337685, 4.132459061661869, 4.1815485732749185, 3.8530777486363244, 3.330100245803114, 4.152316252056483, 4.523012967135754, 4.1166430589995935, 4.770129365109686, 3.427375197819536, 6.552621002391369, 4.1036387387525455, 5.089493601138898, 4.315995322025773, 4.5694606817195815, 3.5136692986556786, 3.3496351999573917, 3.99920512825276, 6.097495504803936, 3.9374828401610977, 4.000875764037219, 3.9337901051546926, 3.9191138368260847, 4.369139212099813, 3.9554418998507326, 4.404489123316532, 4.385601590663141, 3.986249718405627, 3.5190660329683925, 4.3595702386585495, 3.797767544841644, 4.974607963207944, 4.611258051291952, 4.304237008269428, 4.656532361141277, 4.218250446309697, 4.231085600408994, 4.522769775998641, 4.439688239489996, 4.5724907903285406, 4.954622828071775, 4.1857775503531895, 3.607732223467575, 4.242422341310809, 3.9972419961577215]
gammas_mean (on track) 4.279484787586156
success total 78
success 0.78

```


# Next Steps:
- Train even longer --> there is still growth potential!
- Train other two seeds: 3, 7