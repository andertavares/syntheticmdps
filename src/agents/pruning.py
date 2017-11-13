from math import ceil

import numpy as np
import scipy.stats as stats
import agents.uniform as uniform


class PruningAgentWeak(uniform.UniformAgent):
    """
    Only a few actions have nonzero probability, the best action may have zero probability
    """
    def __init__(self, bandit, prune_factor=0.8):
        self.bandit = bandit

        n_arms = len(self.bandit.arms)
        num_nonzero = ceil(n_arms * (1 - prune_factor))

        # at least 2 actions should have non-zero probability
        if num_nonzero < 2:
            num_nonzero = 2

        # determines which actions will have non-zero probability
        which_nonzero = np.random.choice(range(n_arms), num_nonzero, replace=False)

        # samples the prob of choosing arms from U[0, 1]
        probs = stats.uniform(0, 1).rvs(num_nonzero)

        # normalizes
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs]

        # determines probabilities of actions
        self.probabilities = np.zeros(len(bandit.arms))
        for which, prob in zip(which_nonzero, probs):
            self.probabilities[which] = prob

        #checks
        assert np.isclose(1, sum(self.probabilities))

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "PruningAgentWeak"


class PruningAgentStrong(PruningAgentWeak):
    """
    Only a few actions have nonzero probability, the best action (a*) DOES NOT have zero probability
    In fact, it is very likely that p_a* > p_a' for all a' != a*
    """
    def __init__(self, bandit, prune_factor=0.8, p_best_generator = None):
        self.bandit = bandit

        if p_best_generator is None:
            p_best_generator = stats.uniform(0, 1)

        n_arms = len(self.bandit.arms)
        num_nonzero = ceil(n_arms * (1 - prune_factor))

        # at least 2 actions should have non-zero probability
        if num_nonzero < 2:
            num_nonzero = 2

        # determines which actions will have non-zero probability
        which_nonzero = np.random.choice(range(n_arms), num_nonzero, replace=False)

        if bandit.best_arm not in which_nonzero:
            # replaces the first occurrence with the best
            which_nonzero[0] = bandit.best_arm
        else:
            # puts best in first position by swapping
            where = np.where(which_nonzero == bandit.best_arm) # which_nonzero.index(bandit.best_arm)
            which_nonzero[0], which_nonzero[where] =  which_nonzero[where], which_nonzero[0]

        # samples the prob of choosing the best arm from U[0, 1]
        # p_best > all other with high probability
        p_best = 0
        while p_best <= 1.0 / num_nonzero:
            p_best = p_best_generator.rvs(1)[0]

        # ensures p_best <= 1
        p_best = min(1, p_best)

        # samples the prob of choosing the other arms from U[0, 1-p_best]
        p_others = stats.uniform(0, 1 - p_best).rvs(num_nonzero -1)

        # normalizes
        sum_others = sum(p_others)
        p_others = [p *(1 - p_best) / sum_others for p in p_others]

        # if p_best is 1, p_others is zeros
        if np.isclose(p_best, 1):
            p_others = [0] * (num_nonzero - 1)


        # construct a vector with all probabilities (best in first position)
        p_all = [p_best] + p_others

        #print(len(which_nonzero), len(p_all), num_nonzero)

        # determines which actions will have nonzero probability
        self.probabilities = np.zeros(len(bandit.arms))
        for which, prob in zip(which_nonzero, p_all):
            self.probabilities[which] = prob

        #checks
        #print(p_best, sum(self.probabilities))
        assert np.isclose(1, sum(self.probabilities))

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "PruningAgentStrong"

class PruningAgentFair(PruningAgentWeak):
    """
    Only a few actions have nonzero probability, the best action (a*) DOES NOT have zero probability
    It is drawn from a uniform probability, just like the others :)
    """
    def __init__(self, bandit, prune_factor=0.8, p_best_generator = None):
        self.bandit = bandit

        n_arms = len(self.bandit.arms)
        num_nonzero = ceil(n_arms * (1 - prune_factor))

        # at least 2 actions should have non-zero probability
        if num_nonzero < 2:
            num_nonzero = 2

        # determines which actions will have non-zero probability
        which_nonzero = np.random.choice(range(n_arms), num_nonzero, replace=False)

        # ensures the best arm is among non_zero
        if bandit.best_arm not in which_nonzero:
            # replaces the first occurrence with the best
            which_nonzero[0] = bandit.best_arm
        else:
            # puts best in first position by swapping
            where = np.where(which_nonzero == bandit.best_arm) # which_nonzero.index(bandit.best_arm)
            which_nonzero[0], which_nonzero[where] =  which_nonzero[where], which_nonzero[0]

        # samples the prob of choosing arms from U[0, 1]
        probs = stats.uniform(0, 1).rvs(num_nonzero)

        # normalizes
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs]

        # determines probabilities of actions
        self.probabilities = np.zeros(len(bandit.arms))
        for which, prob in zip(which_nonzero, probs):
            self.probabilities[which] = prob

        #checks
        assert np.isclose(1, sum(self.probabilities))
        assert self.probabilities[bandit.best_arm] > 0

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "PruningAgentFair"


class PruningAgentFair2(PruningAgentWeak):
    """
    Only a few actions have nonzero probability, the best action (a*) DOES NOT have zero probability
    It is drawn from a uniform probability, just like the others :)
    """
    def __init__(self, bandit, prune_factor=0.8, u = 1.0, p_best_generator = None):
        self.bandit = bandit

        n_arms = len(self.bandit.arms)
        num_nonzero = ceil(n_arms * (1 - prune_factor))

        # at least 2 actions should have non-zero probability
        if num_nonzero < 2:
            num_nonzero = 2

        # determines which actions will have non-zero probability
        which_nonzero = np.random.choice(range(n_arms), num_nonzero, replace=False)

        # ensures the best arm is among non_zero
        if bandit.best_arm not in which_nonzero:
            # replaces the first occurrence with the best
            which_nonzero[0] = bandit.best_arm
        else:
            # puts best in first position by swapping
            where = np.where(which_nonzero == bandit.best_arm) # which_nonzero.index(bandit.best_arm)
            which_nonzero[0], which_nonzero[where] =  which_nonzero[where], which_nonzero[0]

        # samples the prob of choosing arms from U[0, u]
        prob_best = stats.uniform(0,u).rvs(1)[0];
        probs = stats.uniform(0, 1).rvs(num_nonzero-1)

        # normalizes
        sum_probs = sum(probs)

        factor = sum_probs/(1.0 - prob_best);

        probs = [p / (factor) for p in probs]

        probs.insert(0,prob_best);

        # determines probabilities of actions
        self.probabilities = np.zeros(len(bandit.arms))
        for which, prob in zip(which_nonzero, probs):
            self.probabilities[which] = prob

        #checks
        assert np.isclose(1, sum(self.probabilities))
        assert self.probabilities[bandit.best_arm] > 0

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "PruningAgentFair"
    