
# Todos
## Now

(24.02)

1) Consider direction of landing field (Landeplatzrichtung vs Landeplatzpunkt)
2) - add policy entropy ... --> indicator see if training is going well...

2) Punish for needed time, not distance
3) Try positive rewards again?
- Increase simulation steps after action --> 10, 15 ? even higher?

- Add reward scaling again!
I would suggest that you do not scale by not fitting the reward into a fixed interval 
but by subtracting the mean reward and dividing by the rewards standard deviation. 
If you are building a test set in advance you could calculate those values easily, if you want to do this on the fly, 
you could try to remember the last n rewards and update a running estimate (empirical) of your mean and std which you then use for normalization.
https://www.reddit.com/r/reinforcementlearning/comments/k5bs27/reward_scaling/gehccz3/

- Check how to save best agent in rllib
- Check curiosity: https://www.youtube.com/watch?v=Cc5IZrC_7Ok&ab_channel=anyscale
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...

- Try PPO (GPU version of PPO) instead of TD3 https://openai.com/blog/openai-baselines-ppo/ 

4) Try absolute position of aircraft and target + velocity of aircraft, heading angles of both

- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier?

- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 
- Read DDPG
- Read TD3 paper
- Read PPO paper



## Later
- Steady Wind (Geschwindigkeit und Stärke) --> change on parameter on every episode only

- Train considering engines off and altitude
- Add fix number of obstacles into state (first fix number of obstacles) 
- Train considering wind, engines off and altitude

- Titze: Variable Obstacles 
--> Maybe map image as input to avoid obstacles of current surroundings 
--> Check terrain maps from XPlane


## Maybe
- Höhenabhängiger Wind --> dynamisches Flugmodel

- Decrease max time for episodes (Currently at 5min)
--> 4min
--> 3min
--> 2min
--> 1min

## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Should we give the true heading of the target or really the relative heading of the target to the aircraft