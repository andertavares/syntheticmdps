#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pickle

import config
from agents.learning import LearningAgent
from environment.bandit import Bandit
from experiments.experiment import Experiment, ParallelExperiment
from agents.gaussian import *
from agents.delegator import Delegator
from util import meeting_point

config_obj = config.Config.get_instance()
settings = config_obj.parse(sys.argv[1])


team_sizes = [5] # [5, 10, 15, 20, 25]
bandit_sizes = [10] # [100, 150, 200, 250, 300]
mus = [0.2, 0.4, 0.6]
sigma = 0.2
trials = 5000 #10k
executions = 10 #1000
executionRewardsActions = np.zeros((executions, trials))
executionRewardsGaussian = np.zeros((executions, trials))
p_best_ltd = np.zeros((executions, trials))
p_best_lta = np.zeros((executions, trials))
#expNumber = int(sys.argv[1])

experiments = []
exp_dict = {}

name = "results_gaussian"

for n_arms in bandit_sizes:
    for team_sz in team_sizes:
        for currentMu in mus:
            os.system("mkdir -p " + os.path.join(name, str(n_arms), str(team_sz), '%.2f' % currentMu))

            # identifies groups of experiments by their parameters
            exp_group_name = '%d/%d/%.2f' % (n_arms, team_sz, currentMu)
            exp_dict[exp_group_name] = {'LtA': [], 'LtD': []}

            for e in range(executions):

                bandit = Bandit(n_arms, None, 0.25)

                learner = LearningAgent(bandit, alpha=1.0, epsilon=1.0, alpha_decay=0.999, epsilon_decay=0.999)
                ctrl_gaussian = Delegator(
                    [GaussianAgentPrune(
                        bandit, 0.95, mu=currentMu, sigma=sigma
                    ) for _ in range(team_sz)],
                    alpha=1.0,
                    epsilon=1.0, alpha_decay=0.999, epsilon_decay=0.999
                )
                experiment_id = '%d/%d/%d/%.2f' % (e, n_arms, team_sz, currentMu)

                over_actions = Experiment(bandit, learner, 'LtA/' + experiment_id)
                over_gaussian_agents = Experiment(bandit, ctrl_gaussian, 'LtD/' + experiment_id)
                # over_actions.run(trials)
                # over_gaussian_agents.run(trials)
                experiments.append(over_actions)
                experiments.append(over_gaussian_agents)



print('Setup finished.')
manager = ParallelExperiment(experiments)
manager.run(trials)
print('Experiments finished.')
# organize results

for r in manager.result:
    # group by n_arms, team_sz and currentMu: each exec will be an entry
    index_str, execution_str, n_arms, team_sz, currentMu = r.id.split('/')
    exp_group_name = '%s/%s/%s' % (n_arms, team_sz, currentMu)

    #index_str = 'LtA' if r.agent == 'LearningAgent' else 'LtD'
    exp_dict[exp_group_name][index_str].append(r)

print('Results organized')
for exp_group_name, exp_group in exp_dict.items():
    executionRewardsActions = [exp.rewards for exp in exp_group['LtA']]
    executionRewardsGaussian = [exp.rewards for exp in exp_group['LtD']]

    p_best_lta = [exp.p_best for exp in exp_group['LtA']]
    p_best_ltd = [exp.p_best for exp in exp_group['LtD']]

    times_best_lta = [exp.cumulative_times_best for exp in exp_group['LtA']]
    times_best_ltd = [exp.cumulative_times_best for exp in exp_group['LtD']]

    cumulative_rewards_lta = [exp.cumulative_rewards for exp in exp_group['LtA']]
    cumulative_rewards_ltd = [exp.cumulative_rewards for exp in exp_group['LtD']]

    meetingPoint = meeting_point(np.mean(executionRewardsActions, 0), np.mean(executionRewardsGaussian, 0))

    plt.figure()
    plt.plot(np.mean(executionRewardsActions, 0), label="Actions")
    plt.plot(np.mean(executionRewardsGaussian, 0), label="Gaussian")
    plt.plot(np.convolve(np.mean(executionRewardsActions, 0), np.ones((100,))/100, mode='valid'))
    plt.plot(np.convolve(np.mean(executionRewardsGaussian, 0), np.ones((100,))/100, mode='valid'))
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(name, exp_group_name, "reward.pdf"))
    plt.close()

    plt.figure()
    plt.plot(np.mean(cumulative_rewards_lta, 0), label="Actions")
    plt.plot(np.mean(cumulative_rewards_ltd, 0), label='Gaussian')
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative reward")
    plt.legend()
    # plt.show()  #it does not work
    plt.savefig(os.path.join(name, exp_group_name, "reward_acc.pdf"))

    plt.figure()
    plt.plot(np.mean(p_best_lta, 0), label="Actions")
    plt.plot(np.mean(p_best_ltd, 0), label='Gaussian')
    plt.xlabel("Iteration")
    plt.ylabel("Prob. of best action")
    plt.legend()
    # plt.show()  #it does not work
    plt.savefig(os.path.join(name, exp_group_name, 'pbest_gaussian.pdf'))

    plt.figure()
    plt.plot(np.mean(times_best_lta, 0), label="Actions")
    plt.plot(np.mean(times_best_ltd, 0), label='Gaussian')
    plt.xlabel("Iteration")
    plt.ylabel("#times played best action so far")
    plt.legend()
    # plt.show()  #it does not work
    plt.savefig(os.path.join(name, exp_group_name, "tbest_gaussian.pdf"))

    pickleFile = open(os.path.join(name, exp_group_name, "results.pickle"), "wb")
    pickle.dump([
        np.mean(executionRewardsActions, 0), np.mean(executionRewardsGaussian, 0),
        np.mean(p_best_lta, 0), np.mean(p_best_ltd, 0),
        meetingPoint
    ], pickleFile)
    pickleFile.close()

print('Done')