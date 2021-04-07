# Punish for goal difference
Experiment Date: 31 March 2021, Time: 20:39 
## What is the experiment about
in each step calculate the difference of the current state to the goal state.
Normalize, multiply with -1 and use as punishment on each step.
Give positive reward only when target and correct heading is reached.

No curriculum learning was used

fixed threshold for runway difference to 10°

# corresponding branch 
experiment_shaped_rewards_sin_cos

commit: 

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

        return np.array([
            x,
            y,
            z,
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
                print(">>>>>>>>>>>>>>>> error_deg", runway_angle_error_deg)
                reward_target_reached = 1
                print(f"Episode: {self.episode_counter}, "
                      f"runway angle: {self.runway_angle_deg}, "
                      f"heading: {self.sim.get_heading_true_deg()}")
                self.runway_angle_threshold_deg -= 1
                if self.runway_angle_threshold_deg < self.min_runway_angle_threshold_deg:
                    self.runway_angle_threshold_deg = self.min_runway_angle_threshold_deg
                print("new self.runway_angle_threshold_deg", self.runway_angle_threshold_deg)
            else:
                reward_target_reached = - np.interp(abs(runway_angle_error_deg), [0, 180], [0, 1])
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
Number of episodes: 560


### Description
Does not get to target very often... and does not hit runway angle.



## Conclusion Description
- Seems to add circles to flight... circular movements like sines. See Andrew NG Paper 1991
# Next Steps
-- 