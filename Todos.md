# Todos
## Now
- Normalize values for network from -1 to 1 or at least -10 to 10 ?
- Remove radius 
- Better use Stable Baselines ?: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
- Implement tipps from Stable Baselines
- Simple Training: Train an agent in Google Colab fixed height
(engines on), no wind, no obstacles to get from point A to B.
- Checkout: https://github.com/ray-project/ray/issues/9123
- Construct randomizer for episodes and params for each episode
- Titze: To deacrease the training time
--> try training in a small range first --> to increase getting reward fast... 
- Create Unit Tests for critical methods before or while training + manual tests (While training...)
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