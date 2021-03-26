# Todos
## Now

- Simplify even more
- Curriculum Learning (close to airport, change heading of aircraft step by step...)


- Upload your code
- Share code with Titze

LatLong -> to local reference system:
https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y

- Simplify like kendo said


- Sparse Rewards work well with Ray TD3 for distance and bearing only... [works great... converges after 30 episodes]
- Change again to observation with bearing and distance only for now [works great... converges after 30 episodes]
- Make SB3 TD3 work for that [works great... converges after 20-30 episodes]
- Make SB3 SAC work for that [default params --> seems to work very well...]
- Make HER work with the best of the two above distance only and bearing [not working well]

- Add runway and aircraft heading to the observation
- Try Curriculum Learning --> runway angle adapts to 
- Make HER work with the last point
- Upload and cleanup repo + add Titze

- Make sure there is enough time for turning 
- limit aircraft action

- Try reward shaping
- Read Andrew NGs Paper for reward shaping
-- positive
-- negative

- Complete Code to push..

Try padding the distance at a specific point to concentrate on
https://www.machinecurve.com/index.php/2020/02/07/what-is-padding-in-a-neural-network/

- RUN TD3
- Run PPO
- Compare both

- Implement all from https://andyljones.com/posts/rl-debugging.html and then run again


## Later
- to n-vector first [DONE - klappt ganz gut, braucht etwas länger als bei distance + true bearing]
https://pypi.org/project/nvector/
https://en.wikipedia.org/wiki/N-vector
Checkout https://en.wikipedia.org/wiki/N-vector
https://en.wikipedia.org/wiki/Horizontal_position_representation

- prepare code for jsbsim wrapper and push

- Transform to Cartesian...
lat, long -> x,y https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
https://en.wikipedia.org/wiki/Equirectangular_projection
and Cartesian
Checkout --> https://github.com/geospace-code/pymap3d
https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature
https://heartbeat.fritz.ai/working-with-geospatial-data-in-machine-learning-ad4097c7228d
https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates


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