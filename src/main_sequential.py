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

    expNumber = int(sys.argv[2])
    
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

    # execution rewards
    execution_rwd_lta = np.zeros((executions, trials))
    execution_rwd_ltd = np.zeros((executions, trials))

    # probability of taking the best action
    p_best_lta = np.zeros((executions, trials))
    p_best_ltd = np.zeros((executions, trials))

    # times the best action was taken
    times_best_lta = np.zeros((executions, trials))
    times_best_ltd = np.zeros((executions, trials))

    # cumulative rewards
    cumulative_rewards_lta = np.zeros((executions, trials))
    cumulative_rewards_ltd = np.zeros((executions, trials))

    # cumulative regrets
    cumulative_regret_lta = np.zeros((executions, trials))
    cumulative_regret_ltd = np.zeros((executions, trials))

    # cumulative regrets in expectation
    cumulative_regret_exp_lta = np.zeros((executions, trials))
    cumulative_regret_exp_ltd = np.zeros((executions, trials))
    
    for n_arms in bandit_sizes:
        for team_sz in team_sizes:
            for param in dist_params:

                print('Preparing for %d/%d/%s' % (n_arms, team_sz, param))

                # if experiment is gaussian, param has two values
                mu_or_upper_bound = param if settings['ltd_type'] == 'uniform' else param[0]

                os.system(
                    "mkdir -p " + os.path.join(
                        settings['output_dir'],str(expNumber), str(n_arms), str(team_sz), '%.2f' % mu_or_upper_bound
                    )
                )

                # identifies groups of experiments by their parameters
                exp_group_name = '%d/%d/%.2f' % (n_arms, team_sz, mu_or_upper_bound)

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

                    # creates and runs the experiments
                    lta_experiment = Experiment(bandit, learner, 'LtA/' + experiment_id)
                    lta_experiment.run(trials)
                    
                    ltd_experiment = Experiment(bandit, controller, 'LtD/' + experiment_id)
                    ltd_experiment.run(trials)

                    # extracts data the experiments just performed
                    execution_rwd_lta[e] = lta_experiment.rewards
                    execution_rwd_ltd[e] = ltd_experiment.rewards
                    
                    p_best_lta[e] = lta_experiment.p_best
                    p_best_ltd[e] = ltd_experiment.p_best

                    times_best_lta[e] = lta_experiment.cumulative_times_best
                    times_best_ltd[e] = ltd_experiment.cumulative_times_best

                    cumulative_rewards_lta[e] = lta_experiment.cumulative_rewards
                    cumulative_rewards_ltd[e] = ltd_experiment.cumulative_rewards

                    cumulative_regret_lta[e] = lta_experiment.cumulative_regrets
                    cumulative_regret_ltd[e] = ltd_experiment.cumulative_regrets

                    cumulative_regret_exp_lta[e] = lta_experiment.cumulative_regrets_exp
                    cumulative_regret_exp_ltd[e] = ltd_experiment.cumulative_regrets_exp

                # here all repetitions of a parameter configuration have finished
                # results will be gathered and plotted in the following lines

                # determines the meeting point (where LtA's performance meets LtD) for each criteria:
                # rewards, prob. of best action, times of best action executed, cumulative reward,
                # cumulative regret and expected regret
                meeting_rewards = meeting_point(np.mean(execution_rwd_lta, 0), np.mean(execution_rwd_ltd, 0))
                meeting_pbest = meeting_point(np.mean(p_best_lta, 0), np.mean(p_best_ltd, 0))
                meeting_tbest = meeting_point(np.mean(times_best_lta, 0), np.mean(times_best_ltd, 0))
                meeting_cumulative_reward = meeting_point(np.mean(cumulative_rewards_lta, 0), np.mean(cumulative_rewards_ltd, 0))
                meeting_cumulative_regret = meeting_point(np.mean(cumulative_regret_ltd, 0), np.mean(cumulative_regret_lta, 0))
                meeting_regret_exp = meeting_point(np.mean(cumulative_regret_exp_ltd, 0), np.mean(cumulative_regret_exp_lta, 0))
                
                # plots the data
                plt.figure()
                plt.plot(np.mean(execution_rwd_lta, 0), label="Actions")
                plt.plot(np.mean(execution_rwd_ltd, 0), label="Delegate")
                plt.plot(np.convolve(np.mean(execution_rwd_lta, 0), np.ones((100,))/100, mode='valid'))
                plt.plot(np.convolve(np.mean(execution_rwd_ltd, 0), np.ones((100,))/100, mode='valid'))
                plt.xlabel("Iteration")
                plt.ylabel("Reward")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'],str(expNumber), str(n_arms), str(team_sz), '%.2f' % mu_or_upper_bound,"reward.pdf"))
                plt.close()

                plt.figure()
                plt.plot(np.mean(p_best_lta, 0),color="#1f77b4", label="Actions")
                plt.plot(np.mean(p_best_ltd, 0),color="#ff7f0e", label="Delegate")
                plt.errorbar(range(0, trials, 50), np.mean(p_best_lta,  0)[0:trials:50], yerr=np.std(p_best_lta,  0)[0:trials:50], color="#1f77b4", fmt=".", capsize=3)
                plt.errorbar(range(0, trials, 50), np.mean(p_best_ltd,  0)[0:trials:50], yerr=np.std(p_best_ltd,  0)[0:trials:50], color="#ff7f0e", fmt=".",  capsize=3)
                plt.xlabel("Iteration")
                plt.ylabel(r"$p_{a^*}$")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'], str(expNumber),  str(n_arms),  str(team_sz),  '%.2f' % mu_or_upper_bound, "pbest.pdf"))
                plt.close()

                plt.figure()
                plt.plot(np.mean(times_best_lta,  0), color="#1f77b4", label="Actions")
                plt.plot(np.mean(times_best_ltd,  0), color="#ff7f0e", label="Delegate")
                plt.errorbar(range(0, trials, 50), np.mean(times_best_lta,  0)[0:trials:50], yerr=np.std(times_best_lta, 0)[0:trials:50], color="#1f77b4", fmt=".", capsize=3)
                plt.errorbar(range(0, trials, 50), np.mean(times_best_ltd,  0)[0:trials:50], yerr=np.std(times_best_ltd, 0)[0:trials:50], color="#ff7f0e", fmt=".",  capsize=3)
                plt.xlabel("Iteration")
                plt.ylabel(r"# $a^*$")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'], str(expNumber),  str(n_arms),  str(team_sz),  '%.2f' % mu_or_upper_bound, "timesBest.pdf"))
                plt.close()

                plt.figure()
                plt.plot(np.mean(cumulative_rewards_lta,  0), color="#1f77b4", label="Actions")
                plt.plot(np.mean(cumulative_rewards_ltd,  0), color="#ff7f0e", label="Delegate")
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_rewards_lta,  0)[0:trials:50], yerr=np.std(cumulative_rewards_lta,  0)[0:trials:50], color="#1f77b4", fmt=".", capsize=3)
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_rewards_ltd,  0)[0:trials:50], yerr=np.std(cumulative_rewards_ltd,  0)[0:trials:50], color="#ff7f0e", fmt=".",  capsize=3)
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative reward")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'], str(expNumber), str(n_arms), str(team_sz),
                                         '%.2f' % mu_or_upper_bound, "cumulativeRewards.pdf"))
                plt.close()

                plt.figure()
                plt.plot(np.mean(cumulative_regret_lta, 0), color="#1f77b4", label="Actions")
                plt.plot(np.mean(cumulative_regret_ltd, 0), color="#ff7f0e", label="Delegate")
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_regret_lta, 0)[0:trials:50],
                             yerr=np.std(cumulative_regret_lta, 0)[0:trials:50], color="#1f77b4", fmt=".", capsize=3)
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_regret_ltd, 0)[0:trials:50],
                             yerr=np.std(cumulative_regret_ltd, 0)[0:trials:50], color="#ff7f0e", fmt=".", capsize=3)
                plt.xlabel("Iteration")
                plt.ylabel(r"$\sum $Regret")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'], str(expNumber), str(n_arms), str(team_sz),
                                         '%.2f' % mu_or_upper_bound, "cumulativeRegret.pdf"))
                plt.close()

                plt.figure()
                plt.plot(np.mean(cumulative_regret_exp_lta, 0), color="#1f77b4", label="Actions")
                plt.plot(np.mean(cumulative_regret_exp_ltd, 0), color="#ff7f0e", label="Delegate")
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_regret_exp_lta, 0)[0:trials:50],
                             yerr=np.std(cumulative_regret_exp_lta, 0)[0:trials:50], color="#1f77b4", fmt=".",
                             capsize=3)
                plt.errorbar(range(0, trials, 50), np.mean(cumulative_regret_exp_ltd, 0)[0:trials:50],
                             yerr=np.std(cumulative_regret_exp_ltd, 0)[0:trials:50], color="#ff7f0e", fmt=".",
                             capsize=3)
                plt.xlabel("Iteration")
                plt.ylabel(r"$E(\sum $Regret$)$")
                plt.legend()
                plt.savefig(os.path.join(settings['output_dir'], str(expNumber), str(n_arms), str(team_sz),
                                         '%.2f' % mu_or_upper_bound, "expectedCumulativeRegret.pdf"))
                plt.close()

                # creates the file to dump the results in
                pickle_file = open(
                    os.path.join(
                        settings['output_dir'], str(expNumber), str(n_arms), str(team_sz),
                        '%.2f' % mu_or_upper_bound, "results.pickle"
                    ),
                    "wb"
                )

                # dumps result data to file
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
                    meeting_cumulative_regret, meeting_regret_exp,
                ], pickle_file)
                pickle_file.close()


''' #commenting out because it's not used in this script
def plot(results, output_dir, ltd_type):

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
        plt.close()

        plt.figure()
        plt.plot(np.mean(p_best_lta, 0), label="Actions")
        plt.plot(np.mean(p_best_ltd, 0), label=ltd_name)
        plt.xlabel("Iteration")
        plt.ylabel("Prob. of best action")
        plt.legend()
        # plt.show()  #it does not work
        plt.savefig(os.path.join(output_dir, exp_group_name, 'pbest.pdf'))
        plt.close()

        plt.figure()
        plt.plot(np.mean(times_best_lta, 0), label="Actions")
        plt.plot(np.mean(times_best_ltd, 0), label=ltd_name)
        plt.xlabel("Iteration")
        plt.ylabel("#times played best action so far")
        plt.legend()
        # plt.show()  #it does not work
        plt.savefig(os.path.join(output_dir, exp_group_name, "tbest.pdf"))
        plt.close()

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
'''

if __name__ == '__main__':
    main()

