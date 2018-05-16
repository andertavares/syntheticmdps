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

    # values of the prob. distribution of agent generation
    # it vary in nature if we're dealing with gaussian or uniform
    dist_params = settings['upper_bounds']

    if settings['ltd_type'] == 'gaussian':
        # must use list comprehension otherwise generator is consumed in 1st use
        dist_params = [x for x in itertools.product(settings['mus'], settings['sigmas'])]

    print('Parameters:')
    pprint.PrettyPrinter().pprint(settings)

    # sets up a number of experiments for each variation on bandit size,
    # team size and distribution parameters
    # the number of repetitions for each variation is in the 'executions' variable
    for n_arms in bandit_sizes:
        for team_sz in team_sizes:
            for param in dist_params:

                print('Preparing for %d/%d/%s' % (n_arms, team_sz, param))

                # if experiment is gaussian, param has two values
                mu_or_upper_bound = param if settings['ltd_type'] == 'uniform' else param[0]

                # TODO: make experiments with different sigmas be written to different places
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
        exp_group_name = '%s/%s/%s' % (n_arms, team_sz, currentMu)

        exp_group = exp_dict.get(exp_group_name, {'LtA': [], 'LtD': []})

        # index_str = 'LtA' if r.agent == 'LearningAgent' else 'LtD'
        exp_group[index_str].append(r)
        exp_dict[exp_group_name] = exp_group

    print('Results organized')
    for exp_group_name, exp_group in exp_dict.items():
        # extracts data of training with algorithms and actions
        trials = exp_group['LtA'][0].trials  # TODO check if the number is the same across all experiment in the group
        execution_rwd_lta = [exp.rewards for exp in exp_group['LtA']]
        execution_rwd_ltd = [exp.rewards for exp in exp_group['LtD']]

        p_best_lta = [exp.p_best for exp in exp_group['LtA']]
        p_best_ltd = [exp.p_best for exp in exp_group['LtD']]

        times_best_lta = [exp.cumulative_times_best for exp in exp_group['LtA']]
        times_best_ltd = [exp.cumulative_times_best for exp in exp_group['LtD']]

        cumulative_rewards_lta = [exp.cumulative_rewards for exp in exp_group['LtA']]
        cumulative_rewards_ltd = [exp.cumulative_rewards for exp in exp_group['LtD']]

        cumulative_regret_lta = [exp.cumulative_regrets for exp in exp_group['LtA']]
        cumulative_regret_ltd = [exp.cumulative_regrets for exp in exp_group['LtD']]

        cumulative_regret_exp_lta = [exp.cumulative_regrets_exp for exp in exp_group['LtA']]
        cumulative_regret_exp_ltd = [exp.cumulative_regrets_exp for exp in exp_group['LtD']]

        # calculates the meeting points (where learning over actions starts
        # to outperform learning over algorithms) for various metrics
        meeting_rewards = meeting_point(np.mean(execution_rwd_lta, 0), np.mean(execution_rwd_ltd, 0))
        meeting_pbest = meeting_point(np.mean(p_best_lta, 0), np.mean(p_best_ltd, 0))
        meeting_tbest = meeting_point(np.mean(times_best_lta, 0), np.mean(times_best_ltd, 0))
        meeting_cumulative_reward = meeting_point(np.mean(cumulative_rewards_lta, 0), np.mean(cumulative_rewards_ltd, 0))
        meeting_cumulative_regret = meeting_point(np.mean(cumulative_regret_ltd, 0), np.mean(cumulative_regret_lta, 0))
        meeting_regret_exp = meeting_point(np.mean(cumulative_regret_exp_ltd, 0), np.mean(cumulative_regret_exp_lta, 0))

        ltd_name = ltd_type.capitalize()  # 'Gaussian' if settings['ltd_type'] == 'gaussian' else 'Uniform'

        # plots instantaneous reward
        plt.figure()
        plt.plot(np.mean(execution_rwd_lta, 0), label="Actions")
        plt.plot(np.mean(execution_rwd_ltd, 0), label=ltd_name)
        plt.plot(np.convolve(np.mean(execution_rwd_lta, 0), np.ones((100,))/100, mode='valid'))
        plt.plot(np.convolve(np.mean(execution_rwd_ltd, 0), np.ones((100,)) / 100, mode='valid'))
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, "reward.pdf"))
        plt.close()

        # plots pbest (probability of selecting the best action
        plt.figure()
        plt.plot(np.mean(p_best_lta, 0), color="#1f77b4", label="Actions")
        plt.plot(np.mean(p_best_ltd, 0), color="#ff7f0e", label=ltd_name)
        plt.errorbar(
            range(0, trials, 50), np.mean(p_best_lta, 0)[0:trials:50],
            yerr=np.std(p_best_lta, 0)[0:trials:50],
            color="#1f77b4", fmt=".", capsize=3
        )
        plt.errorbar(
            range(0, trials, 50), np.mean(p_best_ltd, 0)[0:trials:50],
            yerr=np.std(p_best_ltd, 0)[0:trials:50],
            color="#ff7f0e", fmt=".", capsize=3
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"$p_{a^*} (pbest)$")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, 'pbest.pdf'))
        plt.close()

        # plots the number of times the best action has been selected
        plt.figure()
        plt.plot(np.mean(times_best_lta, 0), color="#1f77b4", label="Actions")
        plt.plot(np.mean(times_best_ltd, 0), color="#ff7f0e", label=ltd_name)
        plt.errorbar(
            range(0, trials, 50), np.mean(times_best_lta, 0)[0:trials:50],
            yerr=np.std(times_best_lta, 0)[0:trials:50],
            color="#1f77b4", fmt=".", capsize=3
        )
        plt.errorbar(
            range(0, trials, 50), np.mean(times_best_ltd, 0)[0:trials:50],
            yerr=np.std(times_best_ltd, 0)[0:trials:50],
            color="#ff7f0e", fmt=".", capsize=3
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"# $a^*$ (#times a* was played)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, "timesBest.pdf"))
        plt.close()

        # plots the cumulative reward
        plt.figure()
        plt.plot(np.mean(cumulative_rewards_lta, 0), color="#1f77b4", label="Actions")
        plt.plot(np.mean(cumulative_rewards_ltd, 0), color="#ff7f0e", label=ltd_name)
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_rewards_lta, 0)[0:trials:50],
            yerr=np.std(cumulative_rewards_lta, 0)[0:trials:50],
            color="#1f77b4", fmt=".", capsize=3
        )
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_rewards_ltd, 0)[0:trials:50],
            yerr=np.std(cumulative_rewards_ltd, 0)[0:trials:50],
            color="#ff7f0e", fmt=".", capsize=3
        )
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative reward")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, "cumulativeRewards.pdf"))
        plt.close()

        # plots the cumulative regret
        plt.figure()
        plt.plot(np.mean(cumulative_regret_lta, 0), color="#1f77b4", label="Actions")
        plt.plot(np.mean(cumulative_regret_ltd, 0), color="#ff7f0e", label="Delegate")
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_regret_lta, 0)[0:trials:50],
            yerr=np.std(cumulative_regret_lta, 0)[0:trials:50],
            color="#1f77b4", fmt=".", capsize=3
        )
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_regret_ltd, 0)[0:trials:50],
            yerr=np.std(cumulative_regret_ltd, 0)[0:trials:50],
            color="#ff7f0e", fmt=".", capsize=3
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"$\sum $Regret")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, "cumulativeRegret.pdf"))
        plt.close()

        # plots the expected cumulative regret
        plt.figure()
        plt.plot(np.mean(cumulative_regret_exp_lta, 0), color="#1f77b4", label="Actions")
        plt.plot(np.mean(cumulative_regret_exp_ltd, 0), color="#ff7f0e", label="Delegate")
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_regret_exp_lta, 0)[0:trials:50],
            yerr=np.std(cumulative_regret_exp_lta, 0)[0:trials:50],
            color="#1f77b4", fmt=".", capsize=3
        )
        plt.errorbar(
            range(0, trials, 50), np.mean(cumulative_regret_exp_ltd, 0)[0:trials:50],
            yerr=np.std(cumulative_regret_exp_ltd, 0)[0:trials:50],
            color="#ff7f0e", fmt=".", capsize=3
        )
        plt.xlabel("Iteration")
        plt.ylabel(r"$E(\sum $Regret$)$")
        plt.legend()
        plt.savefig(os.path.join(output_dir, exp_group_name, "expectedCumulativeRegret.pdf"))
        plt.close()

        # dumps the aggregate/mean results to a pickle file
        pickle_file = open(os.path.join(output_dir, exp_group_name, "results.pickle"), "wb")
        pickle.dump([
            np.mean(execution_rwd_lta, 0),
            np.mean(execution_rwd_ltd, 0),
            np.mean(p_best_lta, 0),
            np.mean(p_best_ltd, 0),
            np.mean(times_best_lta, 0),
            np.mean(times_best_ltd, 0),
            np.mean(cumulative_rewards_lta, 0),
            np.mean(cumulative_rewards_ltd, 0),
            np.mean(cumulative_regret_lta, 0),
            np.mean(cumulative_regret_ltd, 0),
            np.mean(cumulative_regret_exp_lta, 0),
            np.mean(cumulative_regret_exp_ltd, 0),
            meeting_rewards, meeting_pbest, meeting_tbest, meeting_cumulative_reward,
            meeting_cumulative_regret, meeting_regret_exp
        ], pickle_file)
        pickle_file.close()


if __name__ == '__main__':
    main()

