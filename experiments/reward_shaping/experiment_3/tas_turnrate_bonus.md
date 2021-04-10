# Punish for goal difference
Experiment Date: 09 April 2021, Time: 20:39 
## What is the experiment about
Added true air speed and turn rate to observation
also a bonus for getting closer to 0 runway angle error on reaching target.

# corresponding branch 
experiment_shaping_turnrate_tas


# Observation function:
```
    def _get_observation(self) -> np.array:
        aircraft_position = self.sim.get_aircraft_geo_position()

        wgs84 = nv.FrameE(name='WGS84')
        pointA = wgs84.GeoPoint(latitude=aircraft_position.lat_deg,
                                longitude=aircraft_position.long_deg,
                                z=self.sim[prp.altitude_sl_ft], degrees=True)
        pointB = wgs84.GeoPoint(latitude=self.target_position.lat_deg,
                                longitude=self.target_position.long_deg,
                                z=self.sim[prp.altitude_sl_ft], degrees=True)
        p_AB_N = pointA.delta_to(pointB)
        x, y, z = p_AB_N.pvector.ravel() / 1000

        runway_angle_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)

        true_airspeed = self.sim.get_true_air_speed()
        yaw_rate = self.sim[prp.r_radps]
        turn_rate = self.sim.get_turn_rate()

        return np.array([
            x,
            y,
            z,
            true_airspeed / 1000,
            turn_rate,
            math.sin(math.radians(runway_angle_error_deg)),
            math.cos(math.radians(runway_angle_error_deg))
        ], dtype=np.float32)
```

# Reward function:
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
                bonus = 1 - np.interp(abs(runway_angle_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
                reward_target_reached = 1 + bonus
                print(f"Episode: {self.episode_counter}, "
                      f"runway angle: {self.runway_angle_deg}, "
                      f"heading: {self.sim.get_heading_true_deg()}")
                self.runway_angle_threshold_deg -= 1
                if self.runway_angle_threshold_deg < self.min_runway_angle_threshold_deg:
                    self.runway_angle_threshold_deg = self.min_runway_angle_threshold_deg
                print("new self.runway_angle_threshold_deg", self.runway_angle_threshold_deg)
            else:
                reward_target_reached = 1 - np.interp(abs(runway_angle_error_deg), [0, 180], [0, 1])
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
Default from rllib


# Corresponding Branch
experiment_reward_shaping_07_04_2021


# Results 
Number of episodes: around 2000


## Conclusion Description
Reaches target fast and often, bounds are not reached very often
--> heading is not correct however... huge error there...

# Next Steps
-- 