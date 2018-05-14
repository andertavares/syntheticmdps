# syntheticmdps
Reinforcement learning (RL) over actions or algorithms in synthetic Markov Decision Processes (MDPs)

# Motivation

Large-scale MDPs are challenging for RL methods, particularly if the number of actions is large because one has to try out all actions at least once to have an idea on their relative quality. To have a precise idea, one has to experiment for a impractically long time.

In some cases, domain-specific heuristics, scripts or algorithms exist and a decision-maker agent can rely on them: at each state, the agent selects an algorithm, which by its turn selects an action to affect the environment. 
The agent's goal is then to find the best algorithm to activate in each state.
It won't be difficult to find the best algorithm if the set of algorithms is small, but the drawback is that the agent is limited by the performance that its algorithms can attain: it won't achieve optimal performane if none of its algorithms is capable of it, even with infinite training time. 

This repository contains code to empirically evaluate the training time vs attainable performance tradeoff. The code allows one to simulate MDPs with various action spaces and test different portfolio sizes (i.e. numbers of algorithms) and specific characteristics of the algorithms: their generation processes and strength-related parameters.

So far, we're working with single-state MDPs, but the results were able to support the following paper: 
A. R. Tavares, S. Anbalagan, L. S. Marcolino and L. Chaimowicz. _Algorithms or Actions? A Study in Large-Scale Reinforcement Learning_. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018), July 2018.

# Running and configuring experiments

## Execution
The experiment scripts come in two flavors: parallel and sequential. 

To understand the differences, keep in mind that each possible combination of parameters is an experiment, and you can run several repetitions of each experiment.
 
In parallel, all repetitions of each experiment are previously loaded. Then they are dispatched to run in parallel.

In some machines, the memory might be insufficient to store all repetitions prior to dispatch. The sequential script remedies this issue: it loads and runs a repetition, freeing up memory for the next. However, it is slower for not running in parallel.

The commands are:

- Parallel - to run experiments making use of parallel processing capabilities, use:
```bash
$ python3 src/main_parallel.py path_to_config_file.ini
```

- Sequential - to run experiments sequentially with low memory consumption, use:
```bash
$ python3 src/main_sequential.py path_to_config_file.ini experiment_identifier
```

Please refer to the configuration file specification in the next section.

## Configuration file

Experiments and repetitions are configured via .ini files. Please find below a sample .ini file with all supported parameters, they're commented in the lines starting with a semicolon:

```ini
;main section, mandatory
[experiment] 

; ltd_type is the algorithm generation process that the agent will use
; LtD means 'Learning to Delegate', an previous naming used on our approach
; Allowed types are: uniform or gaussian
ltd_type = gaussian

; team_sizes is the number of algorithms in the agent's portfolio
; use comma-separated values to test various team sizes
team_sizes = 10,25,50

; bandit_sizes is the number of actions of the underlying problem (a multi-armed bandit)
; use comma-separated values to test various number of actions
bandit_sizes = 100,250,500

; upper_bounds: algorithms generated with uniform have a probability of selecting the best action sampled from U(0, upper_bound)
; This parameter is ignored for gaussian
; use comma-separated values to test various upper_bounds
upper_bounds = 0.1,0.3,0.7

; mus are the means for algorithms generated with gaussian. It is ignored for uniform 
; use comma-separated values to test various mus
mus = 0.1,0.2

; sigmas are the standard deviations for algorithms generated with gaussian. It is ignored for uniform
; use comma-separated values to test various sigmas
sigmas = 0.1,0.3

; trials is the number of interactions with the multi-armed bandit problem.
; The more trials, the higher the chance learning over actions overcomes learning over algorithms
; use comma-separated values to test various number of trials
trials = 1000,5000

; executions is the number of repetitions of each parameter configuration.
; For example, if the parameters are: upper_bounds = 0.1,0.3,0.5; team_sizes = 10,25; 
; bandit_sizes = 100,250; trials = 1000,5000 then there will be 3 x 2 x 2 x 2 = 24 experiments
; if executions = 100, each experiment will be repeated 100 times and averaged, giving 2400 executions in total 
executions = 100

; max_parallel_setup is the maximum number of experiment creation procedures to run in parallel
; currently, it is recommended to leave as 1 because there has been some problems with more 
max_parallel_setup = 1

; max_parallel_runs is the maximum number of experiment execution prodecures to run in parallel.
; It is recommended that this parameter does not exceed the # of cores in your machine
max_parallel_runs = 4

; alpha is the initial learning rate for both agents learning over actions or algorithms
alpha = 1

; aplha_decay is the multiplicative factor by which the initial learning rate is multiplied after each trial on the multi-armed bandit.
; To calculate this value, you can stipulate a desired value for alpha and then use the formula: 
; alpha_decay = (final_alpha / initial_alpha)^ (1/trials) 
; to keep a constant learning rate over all trials, leave this value as 1
; This example value is for decaying alpha from 1 to 0.01 in 5000 trials
alpha_decay = 0.99907939

; epsilon is the initial exploration probability for the learning agents
epsilon = 0.1

; epsilon_decay is the multiplicative factor by which the initial epsilon is multiplied after each trial on the multi-armed bandit. 
; To estimate a proper decay, or to keep epsilon constant, the same rules of alpha_decay apply.
epsilon_decay = 1

; The directory where the experiment results will be written. It is created in a relative path
; based on where the python interpreter is being called. Inside this directory, many subdirectories
; will be created, each with the results of a parameter configuration. 
; For uniform, the directories will look like: bandit_size/team_size/upper_bound for each variation in these values.
; For gaussian, the directories will look like: bandit_size/team_size/mu for each variation in these values.
; Additionally, for main_sequential, the directories will be preceded by the identifier used in command line.
output_dir = results_sample

```
