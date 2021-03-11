
# Todos
## Now

- Code teilen und JSBSIM_Wrapper für Guidance ausrichten und pushen...

- Impelement Stable Baselines again? But how to render images?


- Normalize rewards + ask question in discord / discus.ray...
- RUN TD3
- Run PPO
- Compare both



- Implement all from https://andyljones.com/posts/rl-debugging.html and then run again


## Later
- Train considering engines off and altitude
- Add fix number of obstacles into state (first fix number of obstacles) 

- Parallelization + Check how to save best agent in rllib
- Implement own Algorithm?

- Steady Wind (Geschwindigkeit und Stärke) --> change on parameter on every episode only


- Train considering wind, engines off and altitude

- Titze: Variable Obstacles 
--> Maybe map image as input to avoid obstacles of current surroundings 
--> Check terrain maps from XPlane


## Maybe
- Höhenabhängiger Wind --> dynamisches Flugmodel
- Consider Floyd Hub
4) Try absolute position of aircraft and target + velocity of aircraft, heading angles of both

- Punish for needed time, not distance
- Try positive rewards again?
- Increase simulation steps after action --> 10, 15 ? even higher?

- increase numbers for final rewards (out of bounds, targtet), but add reward scaling again and ...
I would suggest that you do not scale by not fitting the reward into a fixed interval 
but by subtracting the mean reward and dividing by the rewards standard deviation. 
If you are building a test set in advance you could calculate those values easily, if you want to do this on the fly, 
you could try to remember the last n rewards and update a running estimate (empirical) of your mean and std which you then use for normalization.
https://www.reddit.com/r/reinforcementlearning/comments/k5bs27/reward_scaling/gehccz3/

- Decrease max time for episodes (Currently at 5min)
--> 4min
--> 3min
--> 2min
--> 1min

- Check curiosity: https://www.youtube.com/watch?v=Cc5IZrC_7Ok&ab_channel=anyscale
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...

- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 


## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Should we give the true heading of the target or really the relative heading of the target to the aircraft