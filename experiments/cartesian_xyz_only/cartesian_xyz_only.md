# Punish for goal difference
Experiment Date: 15 April 2021, Time: 19:58
## What is the experiment about
Using x,y,z only in observation to get aircraft to target. 

# corresponding branch
experiment_cartesian_xyz_only

#action space 
action_space: gym.Space = gym.spaces.Discrete(36)

# cartesian
```
    def aircraft_cartesian_position(self):
        x = self.sim[prp.dist_from_start_lon_m]
        y = self.sim[prp.dist_from_start_lat_m]

        if self.sim[prp.lat_geod_deg] < self.sim[prp.initial_latitude_geod_deg]:
            y = -y
        if self.sim[prp.long_gc_deg] < self.sim[prp.initial_longitude_geoc_deg]:
            x = -x

        z = self.sim[prp.altitude_sl_m]

        return CartesianPosition(x / 1000, y / 1000, z / 1000)
```

# Observation function (for experiment_sin_cos only)
```
      def _get_observation(self) -> np.array:
        aircraft_position = self.aircraft_cartesian_position()
        diff = self.target_position - aircraft_position

        return np.array([
            diff.x,
            diff.y,
            0
        ], dtype=np.float32)
```

# Reward function 
```
    def _reward(self):
        if self.sim.is_aircraft_out_of_bounds(self.max_distance_km):
            return -2

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM):
            return 1

        aircraft_position = self.aircraft_cartesian_position()
        current_distance_to_target_km = aircraft_position.distance_to_target(self.target_position)
        return -current_distance_to_target_km / 100
```

# Algorithm
## Used Algorithm
TD3
## Used Framework
Rllib
## Algorithm Hyperparams
```
default...
```

# Results
Number of episodes: 300-400
Number of steps:


### Example images in the end of training (10)

### Description

### Graph for all seeds

## Conclusion Description
Does converge at around 250 episodes (quite fast) and pretty good.
Simplified calculations are possible now that we use relative x,y,z cartesian 
# Next Steps
