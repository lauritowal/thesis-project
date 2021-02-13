# Todos
## Now
- Try negative rewards
- Discretesize action ? 360° / 3° ? Or Instead of floats using 360 ints as actions? --> Makes it easier
- Wait few seconds before starting training --> PID controller takes some time in the beginning to stabalize...
- How to make sure that a target is reachable?
- Try the algorithm and the state on a very simplified environment...
- Create Unit Tests for critical methods before or while training + manual tests (While training...) https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/e8p2as5/
- Increase target radius at the beginning a lot --> reduce with time...
- Construct Randomizer for episodes and params for each episode
- Titze: To decrease the training time
--> try training in a small range first --> to increase getting reward fast... 

- Simple Training: Train an agent in Google Colab fixed height
(engines on), no wind, no obstacles to get from point A to B.
- Checkout: https://github.com/ray-project/ray/issues/9123

- Better use Stable Baselines ?: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
- Steady Wind (Geschwindigkeit und Stärke) --> change on parameter on every episode only

## Later
- Consider direction of landing field (Landeplatzrichtung vs Landeplatzpunkt)
- Train considering engines off and altitude
- Add fix number of obstacles into state (first fix number of obstacles) 
- Train considering wind, engines off and altitude

- Titze: Variable Obstacles 
--> Maybe use DQN as map image input to avoid obstacles of current surroundings 
--> Check terrain maps from XPlane


## Maybe
- Höhenabhängiger Wind --> dynamisches Flugmodel

## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Should we give the true heading of the target or really the relative heading of the target to the aircraft