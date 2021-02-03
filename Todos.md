# Todos
## Now
- Work on ORCA: https://stackoverflow.com/questions/55109736/i-use-colab-and-plotly-and-i-want-save-a-plotly-figure-how-to-install-plotly-o
- Simple Training: Train an agent in Google Colab with RLLib for fixed height 
(engines on), no wind, no obstacles to get from point A to B.
- Create Unit Tests for critical methods before or while training + manual tests (While training...)
- Steady Wind (Geschwindigkeit und Stärke) --> change on parameter on every episode only

## Later
- Consider direction of landing field (Landeplatzrichtung vs Landeplatzpunkt)
- Add obstacles into state (first fix number of obstacles) 
- Train considering engines off and altitude
- Train considering wind, engines off and altitude
- Titze: Variable Obstacles --> Maybe use DQN as map image input to avoid obstacles of current surroundings
- Titze: To deacrease the training time--> try training in a small range first --> to increase getting reward fast... 
- Construct randomizer for episodes and params for each episode

## Maybe
- Höhenabhängiger Wind --> dynamisches Flugmodel

## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Should we give the true heading of the target or really the relative heading of the target to the aircraft