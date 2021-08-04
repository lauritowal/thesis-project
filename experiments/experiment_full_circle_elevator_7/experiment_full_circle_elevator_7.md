# Elevator 7 [VEERY GOOD!]
Experiment Date: 14 June 2021, Time: 23:57
## What is the experiment about

Remove penalty for height in each step experiment_full_circle_elevator_7


# corresponding branch
experiment_full_circle_elevator_7

# tensorboard / ray results: 
See google colab: 

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


# Seeds
"seed": 4 (Colab)
TD3_guidance-continuous-v0_2021-06-14_13-46-260fmowqi_

"Seed": 3 (Colab)
TD3_guidance-continuous-v0_2021-07-28_13-48-49dqb89uqy

"Seed": 7 (Colab)
TD3_guidance-continuous-v0_2021-08-03_09-58-56ne15fcfj


### Example images in the end of training (10)

### Description

checkpoint 3501 performs pretty good. See evaluation

Removing the penalty for height seems to make the result better.

# Evaluation:

SEED=3
TD3_guidance-continuous-v0_2021-07-28_13-48-49dqb89uqy (Number of Episodes: ###)
...


TRAINING SEED=4 (Colab):
- TD3_guidance-continuous-v0_2021-06-14_13-46-260fmowqi_
- checkpoint 9901

EVALUATION SEED=1
```
SEED 1
##################################
std_reward 342.21864962839186
mean_reward -24.979426683724242
at target 27
on tracks 79
altitude_rates_mean -17.026277374940776
headings_sum 41
alphas mean 5.255170301763984
ground_speed mean 111.8414779238782
pitches mean -1.2743533955103232
others_sum 70
bounds_sum 3
num total episodes 100
distances_global_mean 0.9398786355578553
tas_mean (fps) 110.81403408195102
distances_on_track_mean 0.5294629610953762
runway_angle_errors (all) 41.79080467354642
runway_angle_errors (on track) 9.654466084742477
all gammas_deg [5.295970731762743, 3.819319955297427, 3.6785459886790175, 3.912522958484683, 2.738402871661972, 3.9561670819265147, 2.8502856565670736, 3.645161604808731, 4.245636981481729, 3.3217716772288597, 3.6191536645817437, 3.8269247925834726, 4.389984027447367, 3.312004342323918, 2.984919736483146, 2.828973485813509, 3.3321485757882736, 4.724125723341783, 3.614567700047851, 2.7640228724997566, 2.638112884827066, 2.9631292834697947, 5.346323601254701, 4.209421534417715, 4.250179784689654, 4.409323346691127, 2.7062794563288444, 3.970631877795302, 3.9022732719512168, 5.001329759317709, 3.8903829013118276, 4.257473502648656, 3.725225277614112, 4.404317448564513, 3.6265781692412355, 3.7027409569947842, 5.098627431002881, 3.9649119157418253, 3.224105094818215, 4.702321091471779, 4.766339972371446, 4.159997059525662, 4.667422185216854, 3.763702982425355, 5.291924823486371, 3.291097160457102, 4.590273355583697, 4.355130603933361, 4.3631312071952415, 3.9831128442557153, 4.055342799695959, 2.821155348659026, 2.4805403866605205, 3.7634181988756, 2.7587713062463193, 4.875462732576465, 4.198545643364976, 3.8637986883246747, 4.204849058112512, 3.47063180901155, 4.730396798464762, 3.3926156300458064, 3.315099742688847, 5.302192516334457, 3.0484418729742933, 3.2312494352569203, 4.805544780092912, 4.143890974630773, 3.9205040363981665, 3.3882179928990612, 3.6959108528251954, 3.903237367375089, 2.9803948378130434, 4.82748706738619, 4.719042057033143, 2.88823418794727, 2.3770984277557003, 5.220952433943875, 4.248019929554455]
gammas_mean (on track) 3.8824997231184923
success total 79
success 0.79

```
EVALUATION SEED=2
```
#################################
SEED 2
##################################
std_reward 427.72264339700973
mean_reward -80.30684487158031
at target 18
on tracks 73
altitude_rates_mean -16.74277929105163
headings_sum 33
alphas mean 5.174923216193557
ground_speed mean 111.78334785332521
pitches mean -1.1452359059651769
others_sum 74
bounds_sum 8
num total episodes 100
distances_global_mean 1.1912377129483516
tas_mean (fps) 110.73723371001147
distances_on_track_mean 0.6679094018992245
runway_angle_errors (all) 47.039743878120554
runway_angle_errors (on track) 10.233336692238822
all gammas_deg [4.03353033907265, 3.7353643521598783, 3.9128016047258707, 3.9483172467935908, 4.775830838003391, 3.9041092892120615, 3.5436209146879287, 3.9560380304536715, 2.803494202060616, 4.1372038569813405, 3.9656653305865426, 4.430761503037848, 4.018051192608011, 3.4368947646917034, 4.136932513468441, 3.556657100633445, 4.412894028422625, 3.727210449460656, 2.4855722363693826, 3.364163057123724, 3.2721036394705965, 3.7219503284467232, 4.049560685475342, 3.902344567381757, 4.480714489347981, 4.164367009290944, 3.6829886909717655, 3.6448505193179637, 5.24386252134467, 2.5582573658917074, 4.7772177915011875, 3.0711430800539836, 4.140086554276662, 4.767314294974619, 3.1852180404207338, 4.165020677641735, 3.6509917072985996, 4.235973156216312, 4.035059236354661, 3.58055007739904, 3.5352281343531913, 3.665869220800699, 3.775076168434863, 3.4891878742347635, 2.9718009050175316, 3.8655561109078893, 3.7808211192204055, 4.6951213960679965, 3.986851708871966, 3.573279656808367, 3.604008318030998, 4.243939653976666, 5.056313646886119, 3.6469297842725688, 4.037831621868289, 3.6386459014847645, 4.637410185735326, 4.088650265760985, 5.358650849306842, 3.5156027036828283, 4.183924663978566, 3.8963088735972042, 4.746081022239405, 2.8163938306734413, 4.456043175234812, 2.889012489258612, 4.111641825153001, 4.929347117055842, 3.4203517576296503, 4.690260440161207, 4.282561027830136, 4.813868788591537, 4.059283228428925, 3.293788355513871]
gammas_mean (on track) 3.9239233662797246
success total 74
success 0.74

```
EVALUATION SEED=3
```
#################################
SEED 3
##################################
std_reward 413.7289944107362
mean_reward -79.40880049180339
at target 17
on tracks 78
altitude_rates_mean -16.55476647542123
headings_sum 45
alphas mean 5.375947490423673
ground_speed mean 111.87692819234184
pitches mean -1.203594459899912
others_sum 80
bounds_sum 3
num total episodes 100
distances_global_mean 0.8951124003907419
tas_mean (fps) 110.98619368247928
distances_on_track_mean 0.49920552272313556
runway_angle_errors (all) 38.80003455479103
runway_angle_errors (on track) 8.418079061933039
all gammas_deg [3.8677353416009064, 3.975544692799529, 4.4377119862412755, 4.399325456764929, 3.720122124530002, 4.077925907300858, 2.414574108358802, 5.254492118373743, 4.3266876406484265, 3.7485017000266607, 4.230854633776546, 5.2631416518745695, 3.8059374977758234, 4.265919872800679, 4.244457427370717, 3.6077779031157524, 4.884704216064835, 4.10782895833566, 3.8349599833996124, 4.131470622484782, 4.4311334426465825, 4.447077895077096, 4.383070688064217, 4.326472685661375, 3.840852538984658, 4.165706533147928, 5.127382166566714, 5.132135108696875, 3.6414368803201294, 4.251409858638723, 3.9392810252478756, 4.685235208220151, 4.219187905230662, 3.3156227756796333, 3.7410761677517317, 3.771552701126833, 4.368508899653561, 3.339017776809708, 4.307009640226946, 4.081174404858931, 4.211367630506784, 3.676847012715752, 4.1823071264177125, 5.12704753162926, 4.267327804434924, 3.2489460127953236, 4.98405004211116, 4.078674815591352, 4.0652514343149635, 4.760756551878579, 4.780431374361497, 3.636129863141823, 4.6562490238005285, 3.007782408813814, 4.106295072146056, 2.5656258844897994, 3.637432974144775, 5.213204457320085, 4.030864703256652, 4.183168261388736, 4.387857254921534, 3.835887378222091, 4.114010790079466, 5.255099975611936, 4.417419120888566, 4.50222543163309, 4.022978858114182, 4.76594839314613, 4.132358172557378, 2.7246939009616713, 4.462188835378841, 3.5901855150540882, 3.8960637091087147, 4.759624141170044, 3.6237442543706173, 4.015622778125652, 3.333106516786508, 4.009069171632945]
gammas_mean (on track) 4.133100799093301
success total 78
success 0.78
```
# Next Steps:
- Train other two seeds: 3, 7
- Try to remove exp for reward again and see if it performs better
- `reward_track = 1 # Maybe diff_track * 2 instead of 1`