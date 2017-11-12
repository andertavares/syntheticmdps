import numpy as np
import uniform
from math import ceil
import scipy.stats as stats


class GaussianAgentPrune(uniform.UniformAgent):
    def __init__(self, bandit, prune_factor=0.8, mu=.5, sigma=.2):
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
            where = np.where(which_nonzero == bandit.best_arm)  # which_nonzero.index(bandit.best_arm)
            which_nonzero[0], which_nonzero[where] = which_nonzero[where], which_nonzero[0]

        # samples the prob of choosing the best arm from N(mu, sigma) - ensuring it is in [0, 1]
        prob_best = stats.norm(mu, sigma).rvs(1)[0]
        prob_best = min(1, max(0, prob_best))

        # samples the prob of choosing the other arms from [0, 1-p_best]
        probs = stats.uniform(0, 1).rvs(num_nonzero - 1)

        sum_probs = sum(probs)

        if (prob_best < 1.0):

            factor = sum_probs / (1.0 - prob_best);

            probs = [p / (factor) for p in probs]

        else:
            probs = [0 for p in probs];

        probs.insert(0, prob_best);

        # determines probabilities of actions
        self.probabilities = np.zeros(len(bandit.arms))
        for which, prob in zip(which_nonzero, probs):
            self.probabilities[which] = prob

        # checks
        # print(p_best, sum(self.probabilities))
        assert np.isclose(1, sum(self.probabilities))

        # saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "GaussianAgent"


class GaussianBiasedAgent(uniform.UniformAgent):
    def __init__(self, bandit, mu=.5, sigma=.2):
        self.bandit = bandit

        # samples the prob of choosing the best arm from N(mu, sigma) - ensuring it is in [0, 1]
        p_best = stats.norm(mu, sigma).rvs(1)[0]
        p_best = min(1, max(0, p_best))


        # samples the prob of choosing the other arms from [0, 1-p_best]
        p_others = stats.uniform(0, 1 - p_best).rvs(len(bandit.arms) -1)

        # normalizes (if needed)
        if not np.isclose(p_best, 1):
            sum_others = sum(p_others)
            #norm_factor = sum_others*(1 - p_best)
            p_others = [p *(1 - p_best) / sum_others for p in p_others]

        #print(p_best, sum_others, sum(p_others))

        # finally assigns the probabilities
        offset = 0 # helps on getting the prob. from correct position
        self.probabilities = np.zeros(len(bandit.arms))
        for i, arm in enumerate(bandit.arms):
            if i == bandit.best_arm:
                self.probabilities[i] = p_best
                offset = 1 # to discount that I'm not getting p_others[i] in this iteration
            else:
                self.probabilities[i] = p_others[i - offset]

        #checks
        #print(p_best, sum(self.probabilities))
        assert np.isclose(1, sum(self.probabilities))

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "GaussianAgent"
