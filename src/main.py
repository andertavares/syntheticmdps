#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pprint
import numpy as np
import pickle
import itertools

import config
from agents.learning import LearningAgent
from environment.bandit import Bandit
from experiments.experiment import Experiment, ParallelExperiment
from agents.gaussian import *
from agents.uniform import *
from agents.pruning import *
from agents.delegator import Delegator
from util import meeting_point


def main():
    config_obj = config.Config.get_instance()
    settings = config_obj.parse(sys.argv[1])

    team_sizes = settings['team_sizes']
    bandit_sizes = settings['bandit_sizes']
    mus = settings['mus']
    sigmas = settings['sigmas']
    trials = settings['trials']
    executions = settings['executions']

    experiments = []
    #exp_dict = {}

    # values of the prob. distribution of agent generation
    # it vary in nature if we're dealing with gaussian or uniform
    dist_params = settings['upper_bounds']

    if settings['ltd_type'] == 'gaussian':
        # must use list comprehension otherwise generator is consumed in 1st use
        dist_params = [x for x in itertools.product(settings['mus'], settings['sigmas'])]

    #name = "results_gaussian"

    print('Parameters:')
    pprint.PrettyPrinter().pprint(settings)

    for n_arms in bandit_sizes:
        for team_sz in team_sizes:
            for param in dist_params:

                print('Preparing for %d/%d/%s' % (n_arms, team_sz, param))

                # if experiment is gaussian, param has two values
                mu_or_upper_bound = param if settings['ltd_type'] == 'uniform' else param[0]

                os.system(
                    "mkdir -p " + os.path.join(
                        settings['output_dir'], str(n_arms), str(team_sz), '%.2f' % mu_or_upper_bound
                    )
                )

                # identifies groups of experiments by their parameters
                exp_group_name = '%d/%d/%.2f' % (n_arms, team_sz, mu_or_upper_bound)

                # will have one experiment for each configuration of the parameters
                experiment_batch = []

                for e in range(executions):
                    sys.stdout.write(
                        '\rSetup for %d arms, |X| = %6d, u/mu = %.4f, exec=%6d' %
                        (n_arms, team_sz, mu_or_upper_bound, e)
                    )

                    bandit = Bandit(n_arms, None, 0.25)

                    learner = LearningAgent(
                        bandit, alpha=settings['alpha'], epsilon=settings['epsilon'],
                        alpha_decay=settings['alpha_decay'], epsilon_decay=settings['epsilon_decay']
                    )

                    if settings['ltd_type'] == 'uniform':
                        controller = Delegator(
                            [PruningAgentFair2(bandit, 0.95, u=mu_or_upper_bound) for _ in range(team_sz)],
                            alpha=settings['alpha'], epsilon=settings['epsilon'],
                            alpha_decay=settings['alpha_decay'], epsilon_decay=settings['epsilon_decay']
                        )
                    else:
                        controller = Delegator(
                            [GaussianAgentPrune(
                                bandit, 0.95, mu=mu_or_upper_bound,  sigma=param[1]
                            ) for _ in range(team_sz)],
                            alpha=settings['alpha'], epsilon=settings['epsilon'],
                            alpha_decay=settings['alpha_decay'], epsilon_decay=settings['epsilon_decay']
                        )

                    experiment_id = '%d/%d/%d/%.2f' % (e, n_arms, team_sz, mu_or_upper_bound)

                    lta_experiment = Experiment(bandit, learner, 'LtA/' + experiment_id)
                    ltd_experiment = Experiment(bandit, controller, 'LtD/' + experiment_id)

                    experiment_batch.append(lta_experiment)
                    experiment_batch.append(ltd_experiment)

                # this batch of experiment is ready. run it:
                print('\nSetup finished for %d experiments.' % len(experiment_batch))
                manager = ParallelExperiment(experiment_batch)
                manager.run(trials)
                plot(manager.result, settings['output_dir'], settings['ltd_type'])
                print('Plot OK for %s' % exp_group_name)


def plot(results, output_dir, ltd_type):

    #exp_dict[exp_group_name] = {'LtA': [], 'LtD': []}
    exp_dict = {}
    print('# results:', len(results))

    for r in results:
        # group by n_arms, team_sz and currentMu: each exec will be an entry
        index_str, execution_str, n_arms, team_sz, currentMu = r.id.split('/')
        #print(index_str, execution_str, n_arms, team_sz, currentMu)
        exp_group_name = '%s/%s/%s' % (n_arms, team_sz, currentMu)
        #print('organizing ', exp_group_name)

        exp_group = exp_dict.get(exp_group_name, {'LtA': [], 'LtD': []})

        # index_str = 'LtA' if r.agent == 'LearningAgent' else 'LtD'
        exp_group[index_str].append(r)
        #print('expgroup', exp_group)
        exp_dict[exp_group_name] = exp_group

    #pprint.PrettyPrinter().pprint(exp_dict)

    print('Results organized')
    for exp_group_name, exp_group in exp_dict.items():
        executionRewardsActions = [exp.rewards for exp in exp_group['LtA']]
        executionRewardsLtD = [exp.rewards for exp in exp_group['LtD']]

        p_best_lta = [exp.p_best for exp in exp_group['LtA']]
        p_best_ltd = [exp.p_best for exp in exp_group['LtD']]

        times_best_lta = [exp.cumulative_times_best for exp in exp_group['LtA']]
        times_best_ltd = [exp.cumulative_times_best for exp in exp_group['LtD']]

        cumulative_rewards_lta = [exp.cumulative_rewards for exp in exp_group['LtA']]
        cumulative_rewards_ltd = [exp.cumulative_rewards for exp in exp_group['LtD']]

        encounter = meeting_point(np.mean(executionRewardsActions, 0), np.mean(executionRewardsLtD, 0))

        ltd_name = ltd_type.capitalize()  # 'Gaussian' if settings['ltd_type'] == 'gaussian' else 'Uniform'

        plt.figure()
        plt.plot(np.mean(executionRewardsActions, 0), label="Actions")
        plt.plot(np.mean(executionRewardsLtD, 0), label=ltd_name)
        plt.plot(np.convolve(np.mean(executionRewardsActions, 0), np.ones((100,))/100, mode='valid'))
        plt.plot(np.convolve(np.mean(executionRewardsLtD, 0), np.ones((100,)) / 100, mode='valid'))
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.legend()

        #print(output_dir, exp_group_name, "reward.pdf")
        plt.savefig(os.path.join(output_dir, exp_group_name, "reward.pdf"))
        plt.close()

        plt.figure()
        plt.plot(np.mean(cumulative_rewards_lta, 0), label="Actions")
        plt.plot(np.mean(cumulative_rewards_ltd, 0), label=ltd_name)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative reward")
        plt.legend()
        # plt.show()  #it does not work
        plt.savefig(os.path.join(output_dir, exp_group_name, "reward_acc.pdf"))

        plt.figure()
        plt.plot(np.mean(p_best_lta, 0), label="Actions")
        plt.plot(np.mean(p_best_ltd, 0), label=ltd_name)
        plt.xlabel("Iteration")
        plt.ylabel("Prob. of best action")
        plt.legend()
        # plt.show()  #it does not work
        plt.savefig(os.path.join(output_dir, exp_group_name, 'pbest.pdf'))

        plt.figure()
        plt.plot(np.mean(times_best_lta, 0), label="Actions")
        plt.plot(np.mean(times_best_ltd, 0), label=ltd_name)
        plt.xlabel("Iteration")
        plt.ylabel("#times played best action so far")
        plt.legend()
        # plt.show()  #it does not work
        plt.savefig(os.path.join(output_dir, exp_group_name, "tbest.pdf"))

        pickleFile = open(os.path.join(output_dir, exp_group_name, "results.pickle"), "wb")
        pickle.dump([
            np.mean(executionRewardsActions, 0),
            np.mean(executionRewardsLtD, 0),
            np.mean(p_best_lta, 0),
            np.mean(p_best_ltd, 0),
            np.mean(cumulative_rewards_lta, 0),
            np.mean(cumulative_rewards_ltd, 0),
            np.mean(times_best_lta, 0),
            np.mean(times_best_ltd, 0),
            encounter
        ], pickleFile)
        pickleFile.close()


if __name__ == '__main__':
    main()

