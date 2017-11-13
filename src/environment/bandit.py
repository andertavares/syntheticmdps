import sys
import config
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool


class Bandit(object):
    def __init__(self, n_arms, initial_mu=None, arms_sigma=1):

        # if not specified, each arm is a gaussian with mean randomly sampled from ~N(0, 1)
        self.bandits_mu = initial_mu if initial_mu is not None else stats.norm(0, 1).rvs(n_arms)

        self.sigma = arms_sigma

        # remembers the best arm and its reward
        self.best_arm = np.argmax(self.bandits_mu)
        self.best_reward = np.max(self.bandits_mu)

        # instantiate the gaussian arms with predefined mu
        config_obj = config.Config.get_instance()
        num_pool = min(config_obj.settings['max_parallel'], n_arms)

        # divides into a number of pools
        self.arms = []

        for i in range(0, n_arms, num_pool):
            sys.stdout.write('\rCreating arm %6d to %6d' % (i, i + num_pool - 1))
            with Pool(num_pool) as p:
                partial_result = p.map(self.new_arm, range(i, i+num_pool))
            self.arms += partial_result
        #self.arms = []

    def new_arm(self, index):
        return stats.norm(self.bandits_mu[index], self.sigma)

    def play(self, arm):
        """
        Returns the reward of playing a given arm
        """
        return np.random.normal(self.bandits_mu[arm], self.sigma)

    def __str__(self):
        return 'MAB(%d)' % len(self.arms)


if __name__ == '__main__':
    # this test is looking good :)
    import numpy as np

    print("Bandit test")
    b = Bandit(200)

    repetitions = 1000

    for idx, arm in enumerate(b.arms):

        outcomes = [b.play(idx) for rep in range(repetitions)]

        print("%.3f: %.3f" % (b.bandits_mu[idx], np.average(outcomes)))

    print("Done")

