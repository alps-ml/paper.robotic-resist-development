## RRT Connect Motion Planner

This is a RRT Motion Planner based on a RRTConnect algorithm implemented in 
 - https://github.com/theteamatx/x-edr-rrtconnect/tree/main

The RRTConnect algorithm is described originally here:

> James J. Kuffner, and Steven M. LaValle. "RRT-connect: An efficient approach to single-query path planning".
> IEEE Conference on Robotics and Automation (ICRA), 2000.

The implementation contains the following variations from the original algorithm:

    1. Greedy sampling of the goal: Every n-th iteration, sample the goal directly. This biases tree growth towards the goal.

    2. Greedy 'extend': Instead of extending one 'epsilon' step towards a new sampled configuration, this implementation will make multiple 'epsilon' steps while the new extended node is not in collision and making progress towards the goal.

    3. Low dispersion quasi-random sequence (Halton Sequence), for a more uniform and deterministic random sampling of the space, based on the following paper:

> Lucas Janson et. al. "Deterministic Sampling-Based Motion Planning: Optimality, Complexity,
and Performance", 2016. https://arxiv.org/abs/1505.00023
