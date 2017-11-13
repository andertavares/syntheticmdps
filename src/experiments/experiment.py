import os
import sys
import copy
import config
import numpy as np
from multiprocessing import Pool
import multiprocessing


class Experiment(object):
    def __init__(self, env, agent, experiment_id=None):
        np.random.seed()
        self.id = experiment_id
        self.env = env
        self.agent = agent
        self.actions = []
        self.rewards = []
        self.p_best = []
        self.cumulative_rewards = []
        self.cumulative_times_best = []
        self.cumulative_reward = 0
        self.cumulative_regret = 0      # actual cumulative regret
        self.cumulative_regret_exp = 0  # expected cumulative regret
        self.freq_best_action = 0       # no. of times the agent select the best arm
        self.trials = 0

    def run(self, trials, verbose=False):
        times_best_action = 0
        self.trials = trials

        for i in range(trials):

            # retrieves an action, makes the player learn and store stats
            action = self.agent.act()
            self.actions.append(action)

            reward = self.env.play(action)
            self.agent.learn(action, reward)

            # registers statistics
            self.rewards.append(reward)
            self.p_best.append(self.agent.p_best(self.env))
            self.cumulative_reward += reward
            self.cumulative_rewards.append(self.cumulative_reward)
            self.cumulative_regret += self.env.best_reward - reward
            self.cumulative_regret_exp += self.env.best_reward - self.env.bandits_mu[action]

            if action == self.env.best_arm:
                times_best_action += 1
            self.cumulative_times_best.append(times_best_action)

            if verbose:
                sys.stdout.write("\rTrial %8d" % i)
                sys.stdout.flush()

            #print('%d, %d, %f' % (i, action, reward))

        self.freq_best_action = times_best_action / trials

    def __str__(self):
        return "%s in %s in %d trials" % (self.agent, self.env, self.trials)


class ParallelExperiment(object):
    """
    Runs experiments in parallel
    """
    def __init__(self, experiments):
        os.system("taskset -p 0xfffff %d" % os.getpid())
        self.experiments = experiments
        self.result = None
        self.pool = None

    def run(self, trials, verbose=False):
        self.trials = trials
        self.verbose = verbose
        config_obj = config.Config.get_instance()
        num_pool = min(config_obj.settings['max_parallel'], len(self.experiments))

        # self.pool = multiprocessing.Pool(num_pool)

        # divides into a number of pools
        self.result = []

        for i in range(0, len(self.experiments), num_pool):
            sys.stdout.write('\rRunning from %8d to %8d out of %8d' % (i, i+num_pool-1, len(self.experiments)))
            with Pool(num_pool) as p:
                partial_result = p.map(self.run_binded, self.experiments[i:i+num_pool])
            self.result += partial_result

    def run_binded(self, experiment):

        experiment.run(self.trials, self.verbose)
        return copy.copy(experiment)
