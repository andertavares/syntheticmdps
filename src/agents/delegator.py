import numpy as np


class Delegator(object):
    def __init__(self, team, alpha=0.1, epsilon=0.1, alpha_decay=0.99, epsilon_decay=0.99, name='Controller'):
        """
        Receives the array of agents
        """
        self.team = team
        self.q = [0 for _ in self.team]
        self.last_choice = None
        self.last_agent_index = None
        self.alpha = alpha
        self.name = name
        self.epsilon = epsilon
        self.alphaDecay = alpha_decay
        self.epsilonDecay = epsilon_decay

    def p_best(self, bandit):
        best_action = bandit.best_arm

        greedy_agent = np.argmax(self.q)

        e = self.epsilon
        p_best = (1 - e) * self.team[greedy_agent].probabilities[best_action] +\
            e / len(self.team) * sum([a.probabilities[best_action] for a in self.team])

        return p_best



    def act(self):
        """
        Epsilon greedy over agents
        # TODO: break ties randomly in argmax
        """

        if np.random.random() < self.epsilon:
            self.last_agent_index =  np.random.randint(0, len(self.q))
        else:
            self.last_agent_index = np.argmax(self.q)

        self.last_choice = self.team[self.last_agent_index].act()

        self.epsilon = self.epsilon*self.epsilonDecay

        return self.last_choice

    def learn(self, choice, reward):
        """
        Q-learning update rule (without states for now)
        """
        a = self.last_agent_index
        self.q[a] = self.q[a] + self.alpha*(reward - self.q[a])

        self.alpha = self.alpha*self.alphaDecay

    def __str__(self):
        return self.name

