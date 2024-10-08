# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:20:15 2024

@author: ccw
"""

import numpy as np


class Casino:

    def __init__(self, ph):
        self.ph = ph  # probability of head
        self.S = np.arange(1, 100)  # 1 ~ 99
        # including terminal state
        self.observation_space = np.arange(100 + 1)  # 0 ~ 100

    def action_space(self, s):
        """Note: In the book, zero is one of the actions, however,
        it's not reasonable to perminantly stay in a state if the optimal
        action is zero"""
        return np.arange(1, min(s, 100 - s) + 1)

    def step(self, s: int, a: int) -> list[(float, int, int)]:
        """return: probability, reward, s' """
        return [(p, 1 if (s + add) == 100 else 0, s + add)
                for add, p in zip((a, -a), (self.ph, 1 - self.ph))]


class Gambler:

    def __init__(self, env, decimal=5):
        self.env = env
        self.V = np.zeros_like(self.env.observation_space, dtype=float)
        self.n_sweeping = 0
        self.dec = decimal  # control the output accuracy

    def q_value(self, s, a):
        e = 0
        for p, r, s_1 in self.env.step(s, a):
            e += p * (r + 1 * self.V[s_1])
        return e

    def q_values(self, s) -> dict[int: float]:
        """Given state s, get all the expected value for each possible action
        """
        return {a: self.q_value(s, a) for a in self.env.action_space(s)}

    def value_update(self):
        self.n_sweeping += 1  # record
        for s in self.env.S:  # 1 ~ 99
            self.V[s] = max(self.q_values(s).values())

    def value_iteration(self):
        dec = self.dec
        while True:
            V_0 = self.V.copy()
            self.value_update()
            # check values unchange
            apprx_diff = self.V.round(dec) - V_0.round(dec)
            values_unchange = (apprx_diff == 0).all()
            if values_unchange:
                break

    def get_values(self):
        return {s: round(self.V[s], self.dec) for s in self.env.S
                if s not in (0, 100)}

    def get_policy(self):
        def greedy_actions(s):
            return [a for a, v in self.q_values(s).items()
                    if round(v, self.dec) == round(self.V[s], self.dec)]
        # tie break by min
        return {s: min(greedy_actions(s)) for s in self.env.S}


if __name__ == '__main__':
    casino = Casino(0.4)
    print(casino.observation_space)
    print(casino.step(50, 10))
    print(casino.step(50, 50))

    agent = Gambler(casino, decimal=5)
    agent.value_iteration()
    agent.value_update()
    agent.get_values()
    agent.n_sweeping
    agent.get_policy()

    casino = Casino(0.55)
    agent = Gambler(casino, decimal=5)
    agent.value_iteration()
    agent.get_values()
    agent.get_policy()
    agent.n_sweeping
