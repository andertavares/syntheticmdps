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

## Execution
The experiment scripts come in two flavors: parallel and sequential.   

Each possible combination of parameters is a repetition, and the experiments can be configured to run several repetitions. 
In parallel, all repetitions of each parameter combination are previously loaded. Then they are dispatched to run in parallel.

In some machines, the memory might be insufficient to store all repetitions prior to dispatch. The sequential script remedies this issue: it loads and runs a repetition, freeing up memory for the next. However, it runs slower for not running in parallel.

The commands are:

- Parallel - to run experiments making use of parallel processing capabilities, use:
```bash
$ python3 src/main_parallel.py path_to_config_file.ini
```

- Sequential - to run experiments sequentially with low memory consumption, use:
```bash
$ python3 src/main_sequential.py path_to_config_file.ini experiment_identifier
```

Please take a look on the specifications of the config. file in the next section.

## Configuration file