# syntheticmdps
Reinforcement learning (RL) over actions or algorithms in synthetic Markov Decision Processes (MDPs)

# Motivation

Large-scale MDPs are challenging for RL methods, particularly if the number of actions is large because one has to try out all actions at least once to have an idea on their relative quality. To have a precise idea, one has to experiment for a impractically long time.

In some cases, domain-specific heuristics, scripts or algorithms exist and a decision-maker agent can rely on them: at each state, the agent selects an algorithm, which by its turn selects an action to affect the environment. 
The agent's goal is then to find the best algorithm to activate in each state.
It won't be difficult to find the best algorithm if the set of algorithms is small, but the drawback is that the agent is limited by the performance that its algorithms can attain: it won't achieve optimal performane if none of its algorithms is capable of it, even with infinite training time. 

This repository contains code to empirically evaluate the training time vs attainable performance tradeoff. The code allows one to simulate MDPs with various action spaces and test different portfolio sizes (i.e. numbers of algorithms) and specific characteristics of the algorithms: their generation processes and strength-related parameters.

So far, we're working with single-state MDPs, but the results were able to support the following paper: 

# Usage
