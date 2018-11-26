# -*- coding: utf -*-

"""The module contains modified QLearning code.

Code taken and modified from:
https://github.com/auimendoza/cs7641-omscs-a4/blob/300168de1c64d3f2babbb48f57dd28549c74388d/mdplib.py

"""

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import mdptoolbox.util as _util
from mdptoolbox.mdp import _computeDimensions
from mdptoolbox.mdp import _MSG_STOP_EPSILON_OPTIMAL_POLICY
from mdptoolbox.mdp import _MSG_STOP_EPSILON_OPTIMAL_VALUE
from mdptoolbox.mdp import _MSG_STOP_MAX_ITER
from mdptoolbox.mdp import _MSG_STOP_UNCHANGING_POLICY
from mdptoolbox.mdp import MDP


class QLearning(MDP):
    """Modified from pymdptoolbox. Add learning rate."""

    def __init__(
        self,
        transitions,
        reward,
        discount,
        n_iter=10000,
        interval=None,
        learning_rate=None,
        explore_interval=100,
        skip_check=False,
    ):

        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.discount = discount
        self.learning_rate = learning_rate

        self.Q = _np.zeros((self.S, self.A))
        self.mean_discrepancy = list()

        self.policies = list()
        self.iterations = list()
        self.elapsedtimes = list()

        self.interval = interval
        self.explore_interval = explore_interval

    def run(self):
        discrepancy = list()

        self.time = _time.time()
        self.starttime = _time.time()

        s, d = _np.random.randint(0, self.S), self.learning_rate
        for n in range(1, self.max_iter + 1):

            if n % self.explore_interval == 0:
                s = _np.random.randint(0, self.S)

            a = _np.random.randint(0, self.A)
            if _np.random.random() < (1 - (1 / (n) ** (1 / 6))):
                a = self.Q[s, :].argmax()

            p_s_new, p, s_new = _np.random.random(), 0, -1
            while p < p_s_new and s_new < (self.S - 1):
                s_new += 1
                p += self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            d = 1 / _math.sqrt(n + 2)
            if self.learning_rate:
                d = self.learning_rate

            futureQ = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = d * futureQ
            self.Q[s, a] = self.Q[s, a] + dQ

            s = s_new

            discrepancy.append(_np.absolute(dQ))

            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

            if self.interval and n % self.interval == 0:
                self.elapsedtimes.append(_time.time() - self.starttime)
                self.policies.append(self.policy)
                self.iterations.append(n)
                self.mean_discrepancy.append(_np.mean(discrepancy))
                discrepancy = list()

        self.time = _time.time() - self.time

        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

        return None
