import matplotlib
# matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
import numpy as np
import pickle
import itertools
import pprint
import util
import config

'''
The purpose of this script is to load saved experiment results (as pickle files) and plot 
the meeting points (where learning over actions starts to outperform learning over algorithms).

Along the x axis, a metric under measurement (team size, #actions, mu/u) will appear. 
In the y axis, the respective point where learning over actions meets learning 
over algorithms.

The script loads a configuration file to find the of the parameter under test, 
as well as the parameters held fixed.
'''


def load_pickle(datapath):
    """
    Loads a pickle file and returns its data.
    The pickle file was expected to be generated with the following data:

    pickle.dump([
        np.mean(execution_rwd_lta, 0), # mean reward per episode when learning over actions
        np.mean(execution_rwd_ltd, 0), # mean reward per episode when learning over algorithms
        np.mean(p_best_lta, 0),     # mean prob. of play best action per episode when learning over actions
        np.mean(p_best_ltd, 0),     # mean prob. of play best action per episode when learning over algorithms
        np.mean(times_best_lta, 0), # number of times the best action was played when learning over actions
        np.mean(times_best_ltd, 0), # number of times the best action was played when learning over algorithms
        np.mean(cumulative_rewards_lta, 0), # mean cumulative reward per episode when learning over actions
        np.mean(cumulative_rewards_ltd, 0), # mean cumulative reward per episode when learning over algorithms
        np.mean(cumulative_regret_lta, 0),  # mean cumulative regret per episode when learning over actions
        np.mean(cumulative_regret_ltd, 0),  # mean cumulative regret per episode when learning over algorithms
        np.mean(cumulative_regret_exp_lta, 0), # mean expected cumulative regret per episode when learning over actions
        np.mean(cumulative_regret_exp_ltd, 0), # mean expected cumulative regret per episode when learning over algorithms
        meeting_rewards, # meeting point of instantaneous reward
        meeting_pbest, # meeting point of the prob. of playing the best action
        meeting_tbest, # meeting point of the number of times the best action was played
        meeting_cumulative_reward, #meeting point of cumulative rewards
        meeting_cumulative_regret, meeting_regret_exp, # meeting points of cumulative and expected cumulative regrets
    ], pickle_file)
    """
    data_file = open(datapath, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    return data


def gather_tau(x_axis, config_path, tau_index, results_root=None):
    """
    Loads several pickle files generated with a experiment, gathers
    the relevant data (indicated by tau_index) and returns it
    :param x_axis: the experiment parameter being varied, must be a valid field in the .ini config file
    :param config_path: path to the configuration file used in the experiment
    :param tau_index: index to extract the relevant meeting point in the pickle
    files generated in each experiment, for example, the index of meeting_cumulative_reward
    is 15, as it is the 16th data saved into pickle (see the documentation of load_pickle)
    :param results_root: root directory of experiment results. If ommitted, will extract from the config file
    :return: a tuple with two lists: the x and y data extracted from the pickle files
    """

    config_obj = config.Config.get_instance()
    settings = config_obj.parse(config_path)

    # values of the prob. distribution of agent generation
    # it vary in nature if we're dealing with gaussian or uniform
    dist_params = settings['upper_bounds']

    if settings['ltd_type'] == 'gaussian':
        # must use list comprehension otherwise generator is consumed in 1st use
        dist_params = [x for x in itertools.product(settings['mus'], settings['sigmas'])]

    # print('Parameters:')
    # pprint.PrettyPrinter(compact=True).pprint(settings)

    y_values = []  # those values will be extracted from the pickle files
    x_values = settings[x_axis]  # variable under measurement (team size, #actions or upper bound/mu)

    for bandit_size in settings['bandit_sizes']:
        for team_size in settings['team_sizes']:
            for param in dist_params:
                mu_or_upper_bound = param if settings['ltd_type'] == 'uniform' else param[0]

                # a single experiment is saved to a path like root/bandit/team/mu_or_u/results.pickle
                # mu_or_upper_bound must have 2 decimals to match the way directories were created
                path = os.path.join(
                    results_root, str(bandit_size), str(team_size),
                    '%.2f' % mu_or_upper_bound, 'results.pickle'
                )  # '%s/%d/%d/%.2f' % (bandit_size, team_size, mu)

                data = load_pickle(path)

                y_values.append(data[tau_index])

    return x_values, y_values


def plot_tau(x_data, y_data, x_label, y_label):
    """
    Performs a simple plotting
    :param x_data: x axis data
    :param y_data: y axis data
    :param x_label: x axis label
    :param y_label: y axis label
    :return:
    """
    plt.figure(figsize=(3.0, 2))
    plt.plot(
        x_data,
        y_data,
    )

    plt.xlabel(x_label, labelpad=0, fontsize=12)
    plt.ylabel(y_label, labelpad=0, fontsize=12)

    plt.show()
    plt.close()


# simple test
if __name__ == '__main__':
    x_axis = 'team_sizes'  # must be a valid field in the .ini config file
    config_path = '../configs/ijcai18/uniform_team-size.ini'
    results_root = '/tmp/uniform_team/0/' # make sure this exists!

    x_data, y_data = gather_tau(x_axis, config_path, 13, results_root) #13 gives meeting_pbest
    plot_tau(x_data, y_data, x_axis, r'$p_{a^*}$')
