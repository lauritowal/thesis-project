# RL tipps 
# planning fallicy
It mostly takes way more samples then you think it will
Even for mujoco environments where there are no images involved it takes mostly 10^5-10^7 steps to learn
# Reward function
your reward function must capture exactly what you want

"In Mujoco Reacher: Since all locations are known, 
reward can be defined as the distance from the end of the arm to the target, plus a small control cost."

reward hacking "a solution which gives more reward by following a goal which was not specified before"

If you explore to much --> you get unsuable data
If you exploit to much --> you get burned in behaviour which isn't optimal

RL --> strong overfitting to environment, does not generalize well 
e.g. it can't play another atari game after having learned a different but similar one

However in navigation  this does work very well, because goals are randomly selected

Deep RL adds more complexity to machine learning because a new dimension is added. --> randomness
The only way to combat that is to do many experiments

A policy that doesn't find good training examples in 
soon will struggle to learn and end up learning nothing at all 

Start as simple as possible and solve this first --> demonstrate the smallest proof of concept first

Create clean rewards:
Everytime you introduce reward shaping you introduce the chance 
that the agent does something which was not really intended. 
--> a non optimal policy which maximizes thw wrong objective

If using space rewards then they should be used a lot --> reward signals that come quick and often
--> the faster a feedback is given after an action via a reward signal
--> the fater the RL agents finds a way to the higher reward

Imitation Learning and Inverse RL are both fields where an agents learns shown reward functions

Running the same algorithm with the same hyper params using 10 different seeds 
and averaging performance over two splits of 5 seeds can lead to learning curves seemingly coming from different statistical distributions
However, in many papers less than 5 seeds are used!
https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457
--> solution is to use more random seeds