import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pickle

from agents.learning import LearningAgent
from env.bandit import Bandit
from experiments.experiment import Experiment, ParallelExperiment
from agents.gaussian import *
from agents.delegator import Delegator
from util import meeting_point

team_sizes = [5, 10, 15, 20, 25]
bandit_sizes = [100, 150, 200, 250, 300]
mus = [0.2, 0.4, 0.6]
sigma = 0.2
trials = 10000
executions = 1000
executionRewardsActions = np.zeros((executions, trials))
executionRewardsGaussian = np.zeros((executions, trials))
p_best_ltd = np.zeros((executions, trials))
p_best_lta = np.zeros((executions, trials))
expNumber = int(sys.argv[1])

experiments = []
exp_dict = {}

name = "results_gaussian"

for n_arms in bandit_sizes:
    for team_sz in team_sizes:
        for currentMu in mus:
            os.system("mkdir -p " + name + "/" + str(expNumber)+"/"+str(n_arms)+"/"+str(team_sz)+"/"+str(currentMu)+"/");

            for e in range(executions):

                bandit = Bandit(n_arms, None, 0.25)

                learner = LearningAgent(bandit, alpha=1.0, epsilon=1.0, alphaDecay=0.999, epsilonDecay=0.999)
                ctrl_gaussian = Delegator(
                    [GaussianAgentPrune(bandit, 0.95, mu=currentMu, sigma=sigma) for _ in range(team_sz)], alpha=1.0,
                    epsilon=1.0, alphaDecay=0.999, epsilonDecay=0.999
                )
                over_actions = Experiment(bandit, learner)
                over_gaussian_agents = Experiment(bandit, ctrl_gaussian)
                # over_actions.run(trials)
                # over_gaussian_agents.run(trials)
                experiments.append(over_actions)
                experiments.append(over_gaussian_agents)

                experiment_id = '%d/%d/%d/%.2f' % (expNumber, n_arms, team_sz, currentMu)

                exp_dict[experiment_id] = (over_actions, over_gaussian_agents)

manager = ParallelExperiment(experiments)
manager.run(trials)

# results = {
#     'gauss': [r for r in manager.result if str(r.agent) == 'Gaussian'],
#     'unif': [r for r in manager.result if str(r.agent) == 'Uniform'],
#     'actions': [over_actions] * len(team_sizes)
# }

# TODO run experiments, identify them and extract results
executionRewardsActions[e] = over_actions.rewards
executionRewardsGaussian[e] = over_gaussian_agents.rewards
p_best_ltd[e] = over_gaussian_agents.p_best
p_best_lta[e] = over_actions.p_best

meetingPoint = meeting_point(np.mean(executionRewardsActions,0),np.mean(executionRewardsGaussian,0));

plt.figure()
plt.plot(np.mean(executionRewardsActions,0),label="Actions")
plt.plot(np.mean(executionRewardsGaussian,0),label="Gaussian")
plt.plot(np.convolve(np.mean(executionRewardsActions,0), np.ones((100,))/100, mode='valid'));
plt.plot(np.convolve(np.mean(executionRewardsGaussian,0), np.ones((100,))/100, mode='valid'));
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.legend()
plt.savefig(name + "/" + str(expNumber)+"/"+str(n_arms)+"/"+str(team_sz)+"/"+str(currentMu)+"/result.pdf");
plt.close()

pickleFile = open(name + "/" + str(expNumber)+"/"+str(n_arms)+"/"+str(team_sz)+"/"+str(currentMu)+"/results.pickle","wb");
pickle.dump([
    np.mean(executionRewardsActions, 0), np.mean(executionRewardsGaussian,0),
    np.mean(p_best_lta, 0), np.mean(p_best_ltd, 0),
    meetingPoint
], pickleFile)
pickleFile.close()
