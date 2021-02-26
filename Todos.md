# Todos
## Now

(24.02)
- Curriculum Learning seems to make problems... Maybe set phases manually... 
- And read: https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457
- Add keep track of target achieved number, and number out of bounds, and number for not achieved target episodes...
--> put images to corresponding different folder maybe too.

- Add reward rescaling again!
I would suggest that you do not scale by not fitting the reward into a fixed interval 
but by subtracting the mean reward and dividing by the rewards standard deviation. 
If you are building a test set in advance you could calculate those values easily, if you want to do this on the fly, 
you could try to remember the last n rewards and update a running estimate of your mean and std which you then use for normalization.
https://www.reddit.com/r/reinforcementlearning/comments/k5bs27/reward_scaling/gehccz3/

- Check again Debug guides from andyjones github and hyperparams for TD3

- Check how to save best agent in rllib

2) Punish for needed time, not distance
7) Try normalization for reward again?
3) Try positive rewards again?
- Increase simulation steps after action --> 10, 15 ? even higher?
1) translate text for RL group
- Decrease max time for episodes (Currently at 5min)
--> 4min
--> 3min
--> 2min
--> 1min

- Try PPO (GPU version of PPO) instead of TD3 https://openai.com/blog/openai-baselines-ppo/ 

4) Try absolute position of aircraft and target + velocity of aircraft, heading angles of both

- Increase target radius at the beginning a lot 
    --> manual [Implemented, Working?: ]
    --> reduce with time... 


- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier?

- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...
- https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/e8p2as5/
- Checkout: https://github.com/ray-project/ray/issues/9123
- Read DDPG
- Read TD3 paper
- Read PPO paper



## Later
- Steady Wind (Geschwindigkeit und Stärke) --> change on parameter on every episode only
- Consider direction of landing field (Landeplatzrichtung vs Landeplatzpunkt)
- Train considering engines off and altitude
- Add fix number of obstacles into state (first fix number of obstacles) 
- Train considering wind, engines off and altitude

- Titze: Variable Obstacles 
--> Maybe map image as input to avoid obstacles of current surroundings 
--> Check terrain maps from XPlane


## Maybe
- Höhenabhängiger Wind --> dynamisches Flugmodel

## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Should we give the true heading of the target or really the relative heading of the target to the aircraft