import numpy as np
import scipy.stats as stats


class Bandit(object):
    def __init__(self, n_arms, initial_mu=None, arms_sigma=1):

        # if not specified, each arm is a gaussian with mean randomly sampled from ~N(0, 1)
        self.bandits_mu = initial_mu if initial_mu is not None else stats.norm(0, 1).rvs(n_arms)

        self.sigma = arms_sigma

        # remembers the best arm and its reward
        self.best_arm = np.argmax(self.bandits_mu)
        self.best_reward = np.max(self.bandits_mu)

        # instantiate the gaussian arms with predefined mu
        self.arms = [stats.norm(mu, arms_sigma) for mu in self.bandits_mu]

    def play(self, arm):
        """
        Returns the reward of playing a given arm
        """
        return np.random.normal(self.bandits_mu[arm], self.sigma)

    def __str__(self):
        return 'MAB(%d)' % len(self.arms)

