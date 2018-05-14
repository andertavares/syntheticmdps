# ijcai18

This directory contains the configuration files used to run the synthetic experiments of the paper:

A. R. Tavares, S. Anbalagan, L. S. Marcolino and L. Chaimowicz. _Algorithms or Actions? A Study in Large-Scale Reinforcement Learning_. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018), July 2018.

In those synthetic experiments, we varied the problem size (number of actions / n_arms), team size (number of algorithms) and upper bound (u, for uniform-ly generated algorithms) or mean (mu, for gaussian-generated algorithms).

When varying a parameter we kept the others fixed at the following values: 100 actions, 25 algorithms, u = 0.5, mu = 0.4. For gaussian, sigma is always fixed at 0.2.
Each experiment has 10000 trials and is repeated 1000 times.

We test the following variations:
```python
upper_bounds = [0.30,0.40,0.50,0.60,0.70,0.80,0.90]
mus = [0.10,0.20,0.30,0.40,0.50,0.60,0.70]
n_arms = [100,150,200,250,300, 350, 400, 450, 500, 550, 600]
team_sizes = [5,15,25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295]
```

The varied parameter under test and the agent generation type is given in the very file name in this directory.
