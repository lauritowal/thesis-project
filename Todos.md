# Todos
## Now

(24.02)
- Add Stable Baselines Agent ?: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
- Check again Debug guides from andyjones github
- PPO (GPU version of PPO) instead of TD3 https://openai.com/blog/openai-baselines-ppo/ 

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

4) Try absolute position of aircraft and target + velocity of aircraft, heading angles of both
6) Curriculum Learning: https://docs.ray.io/en/master/rllib-training.html#example-curriculum-learning

- Increase target radius at the beginning a lot 
    --> manual [Implemented, Working?: ]
    --> reduce with time... 


- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier?

- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...
- https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/e8p2as5/
- Checkout: https://github.com/ray-project/ray/issues/9123




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