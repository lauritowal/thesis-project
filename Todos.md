# Todos
## Now

- Check how to render plot as image
- Simple Training: Train an agent in Google Colab with RLLib for fixed height 
(engines on), no wind, no obstacles to get from point A to B.
- Create Unit Tests for critical methods before training + manual tests (While training...)

## Later
- Add obstacles into state
- Train considering engines off and altitude
- Train considering wind, engines off and altitude

## Minor todos
- Make own gym environment the proper way: https://github.com/openai/gym/blob/master/docs/creating-environments.md
- Simplify code even more and document

## Questions
- Consider direction of landing field ?
- Transform from lat / long to x,y ?
- How to use distances properl instead of lat / long? Does it even matter?
- Should we give the true heading of the target or really the relative heading of the target to the aircraft