# Todos
## Now
- Adapt reward to include heading [Implemented, Working?: ]
- Checkout evaluation of your trained algorithm with rollout [Implemented, Working?: ]
- Check mean rewards for rllib algo on colab?: [Implemented, Working?: ]
https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5
"In principle, this means that you can store, for example, episode return across many episodes, 
and aggregate the results to get the average episode return across stored values." [Implemented, Working?: ]
- try training in a small range first --> to increase getting reward fast... [Implemented, Working?: ]
- add simulation time to prints also [Implemented, Working?: ]

- Penalize for going out of border:
-- fix number in each time step for being outside [Implemented, Working?: ]
-- maybe: higher number on each step if outside of boundaries 
-- short time outside is okay, to much not...

- Decrease max time for episodes (Currently at 5min)
- Unit function for reward
- Increase target radius at the beginning a lot --> reduce with time...
- Try positive reward when reaching target (1) ? 
- Compare to random agent --> tensorboard performance...

- Increase simulation steps after action --> 10, 15 ? even higher?
- Make a unit test for targets in 4-6 fixed places and see how the reward behaves for an random agent and a perfect agent 
- Curriculum Learning with raylib: https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...
- https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/e8p2as5/
- Checkout: https://github.com/ray-project/ray/issues/9123


## Later
- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier?
- Better use Stable Baselines ?: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
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