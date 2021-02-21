# Todos
## Now
-Do stuff from here to use roll out with custom env:
https://github.com/ray-project/ray/issues/6860
https://github.com/ray-project/ray/blob/master/rllib/rollout.py#L33

- aircraft should be close to target, but target shouldn't have initial position if headings are different 
--> confuses the agent... move away a bit

- Try normalization again?

- Punish for needed time not distance

- Try positive rewards again?

- Try absolute position of aircraft and target + velocity of aircraft, heading angles of both
- Remove distance reward? [Implemented, Working?: Seems to make it much (!) better... Check the numbers tomorrow]
- Adapt reward to include heading [Implemented, Working?: ]
- Checkout evaluation of your trained algorithm with rollout [Implemented, Working?: ]
- Check mean rewards for rllib algo on colab?: [Implemented, Working?: ]
https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5
"In principle, this means that you can store, for example, episode return across many episodes, 
and aggregate the results to get the average episode return across stored values." [Implemented, Working?: ]
- try training in a small range first --> to increase getting reward fast... [Implemented, Working?: ]
- add simulation time to prints also [Implemented, Working?: ]

- Try positive reward when reaching target (10) ? [Implemented, Working?: ]

- Increase target radius at the beginning a lot 
    --> manual [Implemented, Working?: ]
    --> reduce with time... 
    
- Penalize for going out of border:
-- fix number in each time step for being outside (-10) [Implemented, Working?: ]
-- maybe: higher number on each step if outside of boundaries 
-- short time outside is okay, to much not...

- Write: Unit function for reward

- PPO2 (GPU version of PPO) instead of TD3 https://openai.com/blog/openai-baselines-ppo/ 

- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier?

- Decrease max time for episodes (Currently at 5min)
--> 4min
--> 3min
--> 2min
--> 1min

- Compare to random agent --> tensorboard performance...

- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 
- Curriculum Learning with raylib: https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...
- https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/e8p2as5/
- Checkout: https://github.com/ray-project/ray/issues/9123

- Increase simulation steps after action --> 10, 15 ? even higher?

- Better use Stable Baselines ?: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html


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