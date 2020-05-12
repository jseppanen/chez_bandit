"""Chez Bandit: multi-armed bandits"""

from collections import defaultdict
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.express as px

from online_stats import OnlineStats


class GaussianEnv:
    """Bandit environment with normally distributed rewards.

    Means and variances are randomized.
    """

    def __init__(self, num_actions: int):
        self.means = np.random.uniform(size=num_actions)
        self.deviations = np.random.uniform(size=num_actions)
        self.regrets = np.max(self.means) - self.means

    def step(self, action: int) -> Tuple[float, float]:
        reward = np.random.normal(self.means[action], self.deviations[action])
        regret = self.regrets[action]
        return reward, regret


class EpsilonGreedy:
    def __init__(self, num_actions: int, eps: float):
        self.num_actions = num_actions
        self.eps = eps
        self.cum_rewards = np.zeros(num_actions)
        self.num_action_taken = np.zeros(num_actions, dtype=int)

    def act(self) -> int:
        if np.random.random() < self.eps:
            return np.random.choice(self.num_actions)
        else:
            avg_rewards = self.cum_rewards / self.num_action_taken.clip(min=1)
            return tb_argmax(avg_rewards)

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward
        self.num_action_taken[action] += 1


class GaussianThompsonSampling:
    """Thompson sampling for Gaussian rewards.

    https://en.wikipedia.org/wiki/Conjugate_prior

    Chapelle, Li
    An Empirical Evaluation of Thompson Sampling
    https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf
    """

    def __init__(self, num_actions: int):
        self.stats = [OnlineStats() for _ in range(num_actions)]
        self.num_actions = num_actions
        self.initial_uniform = 10 * num_actions

    def act(self) -> int:
        if self.initial_uniform:
            self.initial_uniform -= 1
            return self.initial_uniform % self.num_actions
        # sample means from normal-gamma posterior
        mus = [s.mean for s in self.stats]
        nus = [s.n for s in self.stats]
        alphas = np.array([0.5 * s.n for s in self.stats])
        betas = np.array([0.5 * s.m2 for s in self.stats])
        taus = np.random.gamma(alphas, 1.0 / betas)
        sampled_means = np.random.normal(mus, 1.0 / np.sqrt(nus * taus))
        return tb_argmax(sampled_means)

    def update(self, action: int, reward: float):
        self.stats[action].update(reward)


class UCB1:
    """Upper confidence bound bandit.

    Auer, Cesa-Bianchi, Fischer
    Finite-time Analysis of the Multiarmed Bandit Problem
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cum_rewards = np.zeros(num_actions)
        self.num_action_taken = np.zeros(num_actions, dtype=int)
        self.initial_uniform = num_actions

    def act(self) -> int:
        if self.initial_uniform:
            self.initial_uniform -= 1
            return self.initial_uniform % self.num_actions
        avg_rewards = self.cum_rewards / self.num_action_taken.clip(min=1)
        t = self.num_action_taken.sum()
        reward_ucb = avg_rewards + np.sqrt(2 * np.log(t) / self.num_action_taken)
        return tb_argmax(reward_ucb)

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward
        self.num_action_taken[action] += 1


class Exp3:
    """Exponential-weight algorithm for Exploration and Exploitation (Exp3).

    Auer, Cesa-Bianchi, Freund, Schapire
    The non-stochastic multi-armed bandit problem
    https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
    """

    def __init__(self, num_actions: int, gamma: float):
        self.log_weights = np.zeros(num_actions)
        self.num_actions = num_actions
        self.gamma = gamma
        self.initial_uniform = num_actions

    def _action_propensities(self):
        probs = np.exp(log_softmax(self.log_weights))
        probs = (1 - self.gamma) * probs + self.gamma / self.num_actions
        return probs

    def act(self) -> int:
        if self.initial_uniform:
            self.initial_uniform -= 1
            return self.initial_uniform % self.num_actions
        action_probs = self._action_propensities()
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def update(self, action: int, reward: float):
        propensity = self._action_propensities()[action]
        ips_reward = reward / propensity
        self.log_weights[action] += ips_reward * self.gamma / self.num_actions


class Ews:
    """Exponentially weighted stochastic (EWS) bandit.

    hat tip: https://twitter.com/eigenikos/status/1191279528875741185

    Maillard
    APPRENTISSAGE SÉQUENTIEL: Bandits, Statistique et Renforcement
    https://tel.archives-ouvertes.fr/tel-00845410/PDF/thesis_Maillard.pdf
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cum_rewards = np.zeros(num_actions)
        self.num_action_taken = np.zeros(num_actions, dtype=int)

    def act(self) -> int:
        avg_rewards = self.cum_rewards / self.num_action_taken.clip(min=1)
        avg_gaps = np.max(avg_rewards) - avg_rewards
        action_probs = np.exp(log_softmax(-2 * self.num_action_taken * avg_gaps ** 2))
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward
        self.num_action_taken[action] += 1


class BoltzmannExploration:
    """Softmax aka Boltzmann exploration."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cum_rewards = np.zeros(num_actions)
        self.num_action_taken = np.zeros(num_actions, dtype=int)

    def act(self) -> int:
        avg_rewards = self.cum_rewards / self.num_action_taken.clip(min=1)
        action_probs = np.exp(log_softmax(avg_rewards))
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward
        self.num_action_taken[action] += 1


class Bedr:
    """Boltzmann Exploration Done Right.

    Cesa-Bianchi, Gentile, Lugosi, Neu
    Boltzmann Exploration Done Right
    https://papers.nips.cc/paper/7208-boltzmann-exploration-done-right.pdf
    """

    def __init__(self, num_actions: int, exploration_coef: float):
        self.num_actions = num_actions
        self.exploration_coef = exploration_coef
        self.cum_rewards = np.zeros(num_actions)
        self.num_action_taken = np.zeros(num_actions, dtype=int)

    def act(self) -> int:
        avg_rewards = self.cum_rewards / self.num_action_taken.clip(min=1)
        noise = random_gumbel(self.num_actions)
        scores = avg_rewards + self.exploration_coef * (
            noise / np.sqrt(self.num_action_taken.clip(min=1))
        )
        action = tb_argmax(scores)
        return action

    def update(self, action: int, reward: float):
        self.cum_rewards[action] += reward
        self.num_action_taken[action] += 1


def tb_argmax(xs):
    """Tie breaking argmax."""
    ids = np.where(xs == max(xs))[0]
    return np.random.choice(ids)


def log_softmax(logits):
    return logits - np.logaddexp.reduce(logits)


def random_gumbel(size=None):
    """Sample from random Gumbel distribution."""
    return -np.log(-np.log(np.random.uniform(size=size)))


def evaluate(env, agents, horizon):
    cum_regrets = {
        name: np.zeros(horizon)
        for name in agents
    }

    for t in range(horizon):
        for name, agent in agents.items():
            action = agent.act()
            reward, regret = env.step(action)
            agent.update(action, reward)
            cum_regrets[name][t] = cum_regrets[name][t - 1] + regret
    return cum_regrets


num_trials = 10
num_actions = 10
horizon = 100000

cum_regrets = defaultdict(list)
for _ in range(num_trials):
    env = GaussianEnv(num_actions)
    agents = {
        'eps-greedy-1%': EpsilonGreedy(num_actions, .01),
        'eps-greedy-3%': EpsilonGreedy(num_actions, .03),
        'thompson': GaussianThompsonSampling(num_actions),
        'ucb1': UCB1(num_actions),
        'exp3-1%': Exp3(num_actions, 0.01),
        'ews': Ews(num_actions),
        'boltzmann': BoltzmannExploration(num_actions),
        'bedr': Bedr(num_actions, 1.0),
    }
    cr = evaluate(env, agents, horizon)
    for name, cum_regret in cr.items():
        cum_regrets[name].append(cum_regret)

avgs = {
    name: np.mean(x, 0) for name, x in cum_regrets.items()
}

stds = {
    name: np.std(x, 0) for name, x in cum_regrets.items()
}


d = pd.DataFrame(avgs)
d["time"] = np.arange(horizon)
d = d.iloc[::1000]
d =  d.melt(id_vars="time", var_name="agent", value_name="regret")
fig = px.line(d, x="time", y="regret", color="agent",
              title=f"Cumulative regret by agent (average of {num_trials} trials).")
fig.show()
fig.write_image("mab.png")
