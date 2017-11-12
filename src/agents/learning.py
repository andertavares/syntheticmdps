import numpy as np


class LearningAgent(object):
    def __init__(self, bandit, alpha=0.1, epsilon=0.1, alphaDecay = 0.99, epsilonDecay = 0.99):
        self.bandit = bandit
        self.alpha = alpha
        self.q = np.zeros(len(bandit.arms))
        self.last_choice = None
        self.epsilon = epsilon
        self.alphaDecay = alphaDecay
        self.epsilonDecay = epsilonDecay

    def act(self):
        """
        Epsilon greedy over actions
        TODO: break ties arbitrarily in np.argmax
        """
        if np.random.random() < self.epsilon:
            self.last_choice = np.random.randint(0, len(self.q))
        else:
            self.last_choice = np.random.choice(np.flatnonzero(self.q == self.q.max())) #np.argmax(self.q)
            #np.argmax(self.q)

        self.epsilon = self.epsilon*self.epsilonDecay

        return self.last_choice

    def p_best(self, bandit):
        best_action = bandit.best_arm

        greedy = np.argmax(self.q)
        e = self.epsilon

        return (1 - e) if greedy == best_action else e / len(self.q)

    def learn(self, choice, reward):
        """
        Q-learning update rule (without states for now)
        """
        self.q[choice] = self.q[choice] + self.alpha*(reward - self.q[choice])

        self.alpha = self.alpha*self.alphaDecay

    def __str__(self):
        return "LearningAgent"